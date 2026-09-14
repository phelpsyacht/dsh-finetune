/**
 * Shared fixtures for the dsh-finetune test suite.
 */
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

let dir;
export function testDir() {
  dir ??= mkdtempSync(join(tmpdir(), "dsh-finetune-test-"));
  return dir;
}

/** Write a temp JSONL file and return its absolute path. */
export function writeJsonl(name, lines) {
  const path = join(testDir(), name);
  writeFileSync(path, `${lines.join("\n")}\n`, "utf8");
  return path;
}

/** A valid single chat record (well above min_chars). */
export const VALID_CHAT = JSON.stringify({
  messages: [
    { role: "user", content: "什么是梯度下降？" },
    { role: "assistant", content: "梯度下降是一种迭代优化算法。" },
  ],
});

/**
 * A chat dataset with one dirty case per cleaning rule:
 *   1 valid, 2 dedupe, 3 invalid-json, 4 mixed-format, 5 min_chars,
 *   6 repeat_ratio, 7 repeat_ratio, 8 meaningful_ratio, 9 ends_assistant,
 *   10 empty_after_clean, 11 kept (cleaned+merged), 12 kept.
 */
export function chatDirtyLines() {
  return [
    VALID_CHAT,
    VALID_CHAT, // exact duplicate
    "this is not json",
    JSON.stringify({ instruction: "hi", output: "hello" }), // alpaca inside a chat file
    JSON.stringify({ messages: [{ role: "user", content: "ok" }, { role: "assistant", content: "y" }] }),
    JSON.stringify({ messages: [{ role: "user", content: "a".repeat(40) }, { role: "assistant", content: "b".repeat(40) }] }),
    JSON.stringify({ messages: [{ role: "user", content: "哈哈哈哈哈哈哈哈哈哈哈哈" }, { role: "assistant", content: "哈哈哈哈哈哈哈哈哈哈哈哈" }] }),
    JSON.stringify({ messages: [{ role: "user", content: "😀😀😀🔥🔥🔥" }, { role: "assistant", content: "✨✨✨" }] }),
    JSON.stringify({ messages: [{ role: "user", content: "一个问题" }, { role: "assistant", content: "一个回答" }, { role: "user", content: "追问" }] }),
    JSON.stringify({ messages: [{ role: "user", content: "https://example.com/foo/bar" }, { role: "assistant", content: "https://example.com/baz" }] }),
    JSON.stringify({
      messages: [
        { role: "user", content: "看看这个 <b>页面</b> https://example.com/a\u0000\u200B" },
        { role: "user", content: "还有这个 http://example.com/b  " },
        { role: "assistant", content: "好的\n\n\n 已处理  " },
      ],
    }),
    JSON.stringify({ messages: [{ role: "user", content: "1加1等于几？" }, { role: "assistant", content: "等于2，因为1加1就是2。" }] }),
  ];
}

/** Every rule fires exactly once on the chat dirty dataset with defaults. */
export const CHAT_DIRTY_EXPECTED = {
  invalid: 2,
  empty_after_clean: 1,
  dedupe: 1,
  fuzzy_dupe: 0,
  min_chars: 1,
  max_chars: 0,
  repeat_ratio: 2,
  meaningful_ratio: 1,
  ends_assistant: 1,
  ppl: 0,
  lang: 0,
  inst: 0,
  semantic_dupe: 0,
  topic_downsample: 0,
};
