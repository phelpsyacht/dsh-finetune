import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync, readdirSync } from "node:fs";
import { cleanJsonlDataset } from "../lib/clean.js";
import { datasetCleanTool } from "../lib/tools/dataset-clean.js";
import { validateJsonlDataset, validateJsonlLines } from "../lib/dataset.js";
import { writeJsonl, testDir } from "./helpers.mjs";

const tool = datasetCleanTool();

function chatRecord(user, assistant, extra = {}) {
  return JSON.stringify({ messages: [{ role: "user", content: user }, { role: "assistant", content: assistant }], ...extra });
}

// ---------------- full-width → half-width ----------------

test("cleanJsonlDataset: full-width ASCII + space are converted (mapping table, not NFKC)", () => {
  const result = cleanJsonlDataset(`${chatRecord("１＋１＝２？", "等于２。\u3000谢谢")}\n`, { minChars: 0 });
  const record = JSON.parse(result.text.trim());
  assert.equal(record.messages[0].content, "1+1=2?");
  assert.equal(record.messages[1].content, "等于2。 谢谢");
});

test("cleanJsonlDataset: NFKC-style decompositions (①, ﬁ) are NOT touched", () => {
  const result = cleanJsonlDataset(`${chatRecord("第①题 ﬁle", "答案")}\n`, { minChars: 0 });
  const record = JSON.parse(result.text.trim());
  assert.equal(record.messages[0].content, "第①题 ﬁle");
});

test("cleanJsonlDataset: full_width_to_half=false keeps full-width text", () => {
  const result = cleanJsonlDataset(`${chatRecord("１＋１", "２")}\n`, { minChars: 0, fullWidthToHalf: false });
  const record = JSON.parse(result.text.trim());
  assert.equal(record.messages[0].content, "１＋１");
});

// ---------------- PII masking ----------------

test("cleanJsonlDataset: ID card / phone / email are masked with placeholders", () => {
  const result = cleanJsonlDataset(
    `${chatRecord("请联系 13800138000 或 a.b+c@example.com，身份证 11010119900307723X", "好的")}\n`,
    { minChars: 0 },
  );
  const record = JSON.parse(result.text.trim());
  assert.match(record.messages[0].content, /\[PHONE\]/);
  assert.match(record.messages[0].content, /\[EMAIL\]/);
  assert.match(record.messages[0].content, /\[ID\]/);
  assert.doesNotMatch(record.messages[0].content, /13800138000|a\.b\+c@example\.com|11010119900307723X/);
});

test("cleanJsonlDataset: mask_pii=false leaves PII intact", () => {
  const result = cleanJsonlDataset(`${chatRecord("电话 13800138000", "好")}\n`, { minChars: 0, maskPii: false });
  assert.match(result.text, /13800138000/);
});

// ---------------- per-source thresholds ----------------

test("cleanJsonlDataset: per_source overrides thresholds for matching records only", () => {
  const longSentence = "量子计算是一种利用量子力学原理进行信息处理的技术，在密码学与优化问题等领域有广阔的应用前景。";
  const longSentenceOther = "深度学习通过多层神经网络自动学习数据的层次化表示，在图像识别与自然语言处理等领域表现突出。";
  const longCode = longSentence.repeat(300); // ~27.6k chars total: over default 20k, under per-source 40k
  const longPlain = longSentenceOther.repeat(500); // ~47k chars total: over both
  const lines = [
    chatRecord("短", "内容", { source: "code" }), // 3 chars: kept with minChars override 3
    chatRecord(longCode, longCode, { source: "code" }), // ~54k chars: kept with maxChars 40000
    chatRecord("短", "内容"), // no source: dropped min_chars
    chatRecord(longPlain, longPlain), // no source: dropped max_chars
  ];
  const result = cleanJsonlDataset(`${lines.join("\n")}\n`, {
    perSource: [{ match: "code", minChars: 3, maxChars: 40000 }],
  });
  assert.equal(result.keptLines, 2);
  assert.equal(result.byRule.min_chars, 1);
  assert.equal(result.byRule.max_chars, 1);
  const kept = result.text.trim().split("\n").map((line) => JSON.parse(line));
  assert.ok(kept.every((record) => record.source === "code"));
});

// ---------------- SimHash fuzzy dedupe ----------------

const NEAR_DUP_A = chatRecord(
  "量子计算是一种利用量子力学原理进行信息处理的技术。它通过量子比特（qubit）的叠加态和纠缠态来实现并行计算，" +
    "在密码学、药物研发、优化问题等领域有广阔的应用前景。目前各国研究机构正在积极攻关，努力实现容错量子计算的目标。",
  "以上是对量子计算技术的概述。",
);
const NEAR_DUP_B = chatRecord(
  "量子计算是一种利用量子力学原理开展信息处理的技术。它借助量子比特（qubit）的叠加态和纠缠态来实现并行计算，" +
    "在密码学、药物研发、优化问题等领域有广阔的应用前景。目前各国研究机构正在积极攻关，努力实现容错量子计算的目标。",
  "以上是对量子计算技术的概述。",
);

test("cleanJsonlDataset: rewritten near-duplicate is caught by fuzzy_dupe", () => {
  const result = cleanJsonlDataset(`${NEAR_DUP_A}\n${NEAR_DUP_B}\n`);
  assert.equal(result.keptLines, 1);
  assert.equal(result.byRule.fuzzy_dupe, 1);
});

test("cleanJsonlDataset: fuzzy_dedupe=false keeps near-duplicates", () => {
  const result = cleanJsonlDataset(`${NEAR_DUP_A}\n${NEAR_DUP_B}\n`, { fuzzyDedupe: false });
  assert.equal(result.keptLines, 2);
  assert.equal(result.byRule.fuzzy_dupe, 0);
});

test("cleanJsonlDataset: short records are never fuzzy-deduped", () => {
  const lines = [chatRecord("在吗", "在"), chatRecord("在吗", "好的")];
  const result = cleanJsonlDataset(`${lines.join("\n")}\n`, { minChars: 0 });
  assert.equal(result.keptLines, 2);
  assert.equal(result.byRule.fuzzy_dupe, 0);
});

// ---------------- quarantine ----------------

test("cleanJsonlDataset: boundary samples are quarantined within the margin", () => {
  const lines = [
    chatRecord("aaaaaaaaaaabbbbbbbbb", "cdefghijkl"), // 11/20 = 0.55 → boundary (deviation 0.1)
    chatRecord("a".repeat(40), "b".repeat(40)), // repeat ratio 1.0 → far outside margin
  ];
  const result = cleanJsonlDataset(`${lines.join("\n")}\n`, { quarantine: true, quarantineMargin: 0.25 });
  assert.equal(result.byRule.repeat_ratio, 2);
  assert.equal(result.quarantine.count, 1);
  assert.equal(result.quarantine.records[0].rule, "repeat_ratio");
  assert.ok(Math.abs(result.quarantine.records[0].deviation - 0.1) < 1e-6);
});

test("cleanJsonlDataset: quarantine off by default collects nothing", () => {
  const result = cleanJsonlDataset(`${chatRecord("aaaaaaaaaaabbbbbbbbbb", "cdefghijkl")}\n`, {});
  assert.equal(result.quarantine.enabled, false);
  assert.equal(result.quarantine.count, 0);
});

// ---------------- streaming tool ----------------

test("finetune_dataset_clean: streaming tool matches the in-memory engine", async () => {
  const lines = [
    chatRecord("什么是梯度下降？", "梯度下降是一种迭代优化算法。"),
    chatRecord("什么是梯度下降？", "梯度下降是一种迭代优化算法。"), // exact dup
    chatRecord("哈哈".repeat(12), "哈哈".repeat(12)), // repeat spam
    chatRecord("ok", "y"), // min_chars
    chatRecord("题目", "答案", { source: "code" }),
  ];
  const path = writeJsonl("clean-stream.jsonl", lines);
  const inMemory = cleanJsonlDataset(`${lines.join("\n")}\n`);
  const streamed = await tool.execute({ path, min_chars: 10 }, {});
  assert.equal(streamed.keptLines, inMemory.keptLines);
  assert.equal(streamed.removedLines, inMemory.removedLines);
  assert.deepEqual(streamed.byRule, inMemory.byRule);
  assert.equal(streamed.outputValid, true);
  assert.equal(readFileSync(path, "utf8"), `${lines.join("\n")}\n`); // input untouched
  assert.equal(validateJsonlDataset(readFileSync(streamed.outputPath, "utf8")).valid, true);
});

test("finetune_dataset_clean: dry_run + quarantine writes review files but no output", async () => {
  const lines = [chatRecord("aaaaaaaaaaabbbbbbbbbb", "cdefghijkl"), chatRecord("什么是梯度下降？", "梯度下降是一种迭代优化算法。")];
  const path = writeJsonl("clean-dry-quarantine.jsonl", lines);
  const result = await tool.execute({ path, dry_run: true, quarantine: true }, {});
  assert.equal(result.dryRun, true);
  assert.equal(result.outputPath, "");
  assert.equal(result.quarantine, true);
  assert.equal(existsSync(`${path}.clean.jsonl`), false);
  assert.equal(existsSync(`${path}.quarantine.jsonl`), true);
  assert.equal(existsSync(`${path}.quarantine.html`), true);
  assert.equal(existsSync(`${path}.quarantine.csv`), true);
  const quarantined = JSON.parse(readFileSync(`${path}.quarantine.jsonl`, "utf8").trim());
  assert.equal(quarantined.rule, "repeat_ratio");
  assert.match(readFileSync(`${path}.quarantine.html`, "utf8"), /边界样本/);
  assert.match(readFileSync(`${path}.quarantine.csv`, "utf8"), /line,rule,source/);
});

test("finetune_dataset_clean: PII masking flows through the tool", async () => {
  const path = writeJsonl("clean-pii.jsonl", [chatRecord("电话 13800138000", "好的")]);
  const result = await tool.execute({ path, min_chars: 0 }, {});
  assert.match(readFileSync(result.outputPath, "utf8"), /\[PHONE\]/);
  assert.doesNotMatch(readFileSync(result.outputPath, "utf8"), /13800138000/);
});

// ---------------- streaming validation ----------------

test("validateJsonlLines: streaming result matches validateJsonlDataset", async () => {
  const text = `${chatRecord("a", "b")}\nthis is not json\n${chatRecord("c", "d")}\n`;
  const asyncLines = (async function* () {
    for (const line of text.split("\n")) yield line;
  })();
  assert.deepEqual(await validateJsonlLines(asyncLines), validateJsonlDataset(text));
});

// ---------------- cleanup ----------------

test.after(() => {
  const dir = testDir();
  for (const name of readdirSync(dir)) {
    // leave the sandbox dir in place; the OS temp cleaner owns it
    void name;
  }
});
