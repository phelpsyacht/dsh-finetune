import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { cleanJsonlDataset } from "../lib/clean.js";
import { datasetCleanTool } from "../lib/tools/dataset-clean.js";
import { validateJsonlDataset } from "../lib/dataset.js";
import { writeJsonl, chatDirtyLines, CHAT_DIRTY_EXPECTED } from "./helpers.mjs";

// ---------------- engine ----------------

test("cleanJsonlDataset: every rule fires exactly once on the dirty chat dataset", () => {
  const result = cleanJsonlDataset(`${chatDirtyLines().join("\n")}\n`);
  assert.deepEqual(result.byRule, CHAT_DIRTY_EXPECTED);
  assert.equal(result.inputLines, 12);
  assert.equal(result.keptLines, 3);
  assert.equal(result.removedLines, 9);
  assert.equal(result.format, "chat");
  // each removed line is reported with its original line number
  assert.deepEqual(
    result.examples.map((e) => e.rule).sort(),
    ["dedupe", "empty_after_clean", "ends_assistant", "invalid", "invalid", "meaningful_ratio", "min_chars", "repeat_ratio", "repeat_ratio"],
  );
  assert.equal(result.examples[0].line, 2); // dedupe is the first kept-fail
});

test("cleanJsonlDataset: output is clean and still validates as chat", () => {
  const result = cleanJsonlDataset(`${chatDirtyLines().join("\n")}\n`);
  assert.equal(validateJsonlDataset(result.text).valid, true);
  const records = result.text.trim().split("\n").map((line) => JSON.parse(line));
  // line 11 fixture: HTML/URL/control chars stripped, same-role turns merged
  const merged = records.find((r) => r.messages[0].content.includes("看看这个"));
  assert.ok(merged, "cleaned+merged record missing");
  assert.equal(merged.messages.length, 2);
  assert.equal(merged.messages[0].role, "user");
  assert.equal(merged.messages[0].content, "看看这个 页面\n还有这个");
  assert.ok(!/https?:|<\/?[a-z]|\u0000|\u200B/.test(JSON.stringify(merged)));
});

test("cleanJsonlDataset: short single-char answers are not repetition spam", () => {
  const result = cleanJsonlDataset(
    `${JSON.stringify({ messages: [{ role: "user", content: "在吗" }, { role: "assistant", content: "在" }] })}\n`,
    { minChars: 0 },
  );
  assert.equal(result.keptLines, 1);
  assert.equal(result.byRule.repeat_ratio, 0);
});

test("cleanJsonlDataset: undefined options never disable defaults", () => {
  const result = cleanJsonlDataset(`${chatDirtyLines().join("\n")}\n`, { minChars: undefined, maxChars: undefined });
  assert.deepEqual(result.byRule, CHAT_DIRTY_EXPECTED);
});

test("cleanJsonlDataset: 0 thresholds disable length limits", () => {
  const result = cleanJsonlDataset(`${chatDirtyLines().join("\n")}\n`, { minChars: 0, maxChars: 0 });
  assert.equal(result.byRule.min_chars, 0);
  assert.equal(result.byRule.max_chars, 0);
  assert.equal(result.keptLines, 4); // adds the ok/y record which was min_chars before
});

test("cleanJsonlDataset: empty input stays empty", () => {
  const result = cleanJsonlDataset("");
  assert.equal(result.inputLines, 0);
  assert.equal(result.format, "empty");
  assert.equal(result.text, "");
});

test("cleanJsonlDataset: prompt-completion URL prompt becomes empty_after_clean", () => {
  const result = cleanJsonlDataset(
    `${JSON.stringify({ prompt: "https://example.com", completion: "x" })}\n${JSON.stringify({ prompt: "2+2=?", completion: "4" })}\n`,
    { minChars: 0 },
  );
  assert.equal(result.keptLines, 1);
  assert.equal(result.byRule.empty_after_clean, 1);
});

// ---------------- tool ----------------

const tool = datasetCleanTool();

test("finetune_dataset_clean: writes <path>.clean.jsonl by default, input untouched", async () => {
  const path = writeJsonl("clean-default.jsonl", chatDirtyLines());
  const before = readFileSync(path, "utf8");
  const result = await tool.execute({ path }, {});
  assert.equal(result.outputPath, `${path}.clean.jsonl`);
  assert.equal(result.dryRun, false);
  assert.equal(readFileSync(path, "utf8"), before); // input never modified
  assert.equal(validateJsonlDataset(readFileSync(result.outputPath, "utf8")).valid, true);
});

// Regression: with the optional Python layer disabled (the default), the tool returns
// pythonReport === null. The output schema must accept that, or every default clean
// call fails result validation with `"value.pythonReport" must be an object`.
test("finetune_dataset_clean: default run returns null pythonReport and schema allows null", async () => {
  const path = writeJsonl("clean-null-report.jsonl", chatDirtyLines());
  const result = await tool.execute({ path }, {});
  assert.equal(result.pythonReport, null);
  assert.equal(result.pythonReportPath, "");
  const field = tool.output.schema.properties.pythonReport;
  assert.ok(
    field.oneOf?.some((branch) => branch.type === "null"),
    "pythonReport schema must allow null (Python layer off by default)",
  );
});

test("finetune_dataset_clean: dry_run reports but writes nothing", async () => {
  const path = writeJsonl("clean-dry.jsonl", chatDirtyLines());
  const result = await tool.execute({ path, out: `${path}.dry.jsonl`, dry_run: true }, {});
  assert.equal(result.dryRun, true);
  assert.equal(result.outputPath, "");
  assert.equal(result.removedLines, 9);
  assert.equal(existsSync(`${path}.dry.jsonl`), false);
});

test("finetune_dataset_clean: refuses to overwrite the input file", async () => {
  const path = writeJsonl("clean-overwrite.jsonl", chatDirtyLines());
  await assert.rejects(() => tool.execute({ path, out: path }, {}), /overwrite/i);
});

test("finetune_dataset_clean: honors rule toggles through args", async () => {
  const path = writeJsonl("clean-toggles.jsonl", chatDirtyLines());
  const result = await tool.execute({ path, dry_run: true, dedupe: false, clean_text: false, require_assistant_end: false, min_chars: 0 }, {});
  assert.equal(result.byRule.dedupe, 0);
  assert.equal(result.byRule.ends_assistant, 0);
  assert.equal(result.byRule.min_chars, 0);
});

test("finetune_dataset_clean: rejects missing path and non-integer min_chars", async () => {
  await assert.rejects(() => tool.execute({}, {}), /required/);
  await assert.rejects(() => tool.execute({ path: "/tmp/x.jsonl", min_chars: "ten" }, {}), /type|min_chars/);
});

test("finetune_dataset_clean: sharegpt non-assistant-ending records are dropped as invalid", async () => {
  const path = writeJsonl("clean-sharegpt.jsonl", [
    JSON.stringify({ conversations: [{ from: "system", value: "你是一位乐于助人的助手" }, { from: "human", value: "你好，请介绍一下你自己" }, { from: "gpt", value: "你好！我是智能助手。" }] }),
    JSON.stringify({ conversations: [{ from: "human", value: "hi" }, { from: "gpt", value: "hello" }, { from: "human", value: "again" }] }),
    JSON.stringify({ conversations: [{ from: "human", value: "第一个问题" }, { from: "user", value: "第二个问题" }, { from: "assistant", value: "这是回答内容" }] }),
  ]);
  const result = await tool.execute({ path }, {});
  assert.equal(result.keptLines, 2);
  assert.equal(result.byRule.invalid, 1);
  const kept = readFileSync(result.outputPath, "utf8").trim().split("\n").map((line) => JSON.parse(line));
  assert.equal(kept[1].conversations[0].value, "第一个问题\n第二个问题"); // human+user merged
});
