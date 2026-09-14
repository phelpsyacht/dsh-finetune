import { test } from "node:test";
import assert from "node:assert/strict";
import { validateJsonlDataset } from "../lib/dataset.js";
import { datasetValidateTool } from "../lib/tools/dataset-validate.js";
import { writeJsonl } from "./helpers.mjs";

const CHAT = JSON.stringify({
  messages: [
    { role: "user", content: "什么是梯度下降？" },
    { role: "assistant", content: "梯度下降是一种迭代优化算法。" },
  ],
});
const PROMPT_COMPLETION = JSON.stringify({ prompt: "1+1=?", completion: "2" });
const ALPACA = JSON.stringify({ instruction: "翻译成英文", input: null, output: "Translate to English." });
const SHAREGPT = JSON.stringify({
  conversations: [
    { from: "human", value: "你好" },
    { from: "gpt", value: "你好！" },
  ],
});

test("validateJsonlDataset accepts chat / prompt-completion / alpaca / sharegpt", () => {
  for (const [line, format] of [
    [CHAT, "chat"],
    [PROMPT_COMPLETION, "prompt-completion"],
    [ALPACA, "alpaca"],
    [SHAREGPT, "sharegpt"],
  ]) {
    const result = validateJsonlDataset(`${line}\n`);
    assert.equal(result.valid, true, `expected ${format} to validate`);
    assert.equal(result.format, format);
    assert.equal(result.lines, 1);
    assert.equal(result.errors.length, 0);
  }
});

test("validateJsonlDataset rejects invalid JSON, unknown formats and mixed formats", () => {
  assert.equal(validateJsonlDataset("not json\n").valid, false);
  assert.equal(validateJsonlDataset(`${JSON.stringify({ foo: "bar" })}\n`).valid, false);
  assert.equal(validateJsonlDataset(`${CHAT}\n${ALPACA}\n`).valid, false);
  const errors = validateJsonlDataset(`${CHAT}\n${ALPACA}\n`).errors;
  assert.equal(errors[0].line, 2);
});

test("validateJsonlDataset reports line-accurate errors and caps them at 10", () => {
  const result = validateJsonlDataset(["bad", "worse", "x", "y", "z", "a", "b", "c", "d", "e", "f"].map((x) => x).join("\n"));
  assert.equal(result.valid, false);
  assert.equal(result.errors.length, 10);
  assert.equal(result.errors[0].line, 1);
});

test("validateJsonlDataset treats empty / blank-only files as empty", () => {
  const result = validateJsonlDataset("\n\n  \n");
  assert.equal(result.format, "empty");
  assert.equal(result.lines, 0);
  assert.equal(result.valid, false);
});

test("validateJsonlDataset estimates tokens as chars/4", () => {
  const result = validateJsonlDataset(`${CHAT}\n`);
  assert.equal(result.approxTokens, Math.round(CHAT.length / 4));
});

test("finetune_dataset_validate tool executes end to end on a real file", async () => {
  const path = writeJsonl("validate-ok.jsonl", [CHAT]);
  const result = await datasetValidateTool().execute({ path }, {});
  assert.equal(result.valid, true);
  assert.equal(result.format, "chat");
  assert.equal(result.lines, 1);
});

test("finetune_dataset_validate tool rejects bad files and missing args", async () => {
  const bad = writeJsonl("validate-bad.jsonl", ["not json"]);
  const result = await datasetValidateTool().execute({ path: bad }, {});
  assert.equal(result.valid, false);
  assert.equal(result.errors[0].line, 1);
  await assert.rejects(() => datasetValidateTool().execute({}, {}), /required/);
});
