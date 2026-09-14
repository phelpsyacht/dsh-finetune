import { createReadStream } from "node:fs";
import { createInterface } from "node:readline";
import { defineTool } from "@deepseek-ai/dsh-tools";
import { validateJsonlLines } from "../dataset.js";

const RESULT_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    valid: { type: "boolean" },
    lines: { type: "integer" },
    format: { type: "string" },
    approxTokens: { type: "integer" },
    errors: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        properties: { line: { type: "integer" }, message: { type: "string" } },
      },
    },
    preview: { type: "string" },
  },
};

export function datasetValidateTool() {
  return defineTool({
    name: "finetune_dataset_validate",
    description:
      "Validate a local JSONL fine-tuning dataset before training/upload (read-only). Supported record formats: " +
      'chat ({"messages":[{"role","content"},...]}), prompt/completion, alpaca ({"instruction","output"}), ' +
      'and sharegpt ({"conversations":[{"from","value"},...]}). Mixing formats is rejected. ' +
      "Reports line count, format, approximate token count, and the first per-line errors. Always run this before finetune_job_create or finetune_train.",
    parameters: {
      path: { type: "string", required: true, description: "Absolute path to the .jsonl dataset file" },
    },
    output: {
      schema: RESULT_SCHEMA,
      render: (_args, value) => [
        {
          type: "text",
          text: value.valid
            ? `dataset OK: ${value.lines} ${value.format} records, ~${value.approxTokens} tokens`
            : `dataset INVALID (${value.lines} lines, format ${value.format}): ${value.errors
                .slice(0, 3)
                .map((error) => `line ${error.line}: ${error.message}`)
                .join("; ")}`,
        },
      ],
    },
    isConcurrencySafe: () => true,
    timeoutMs: 30000,
    async execute(args, exec) {
      // Stream line by line so multi-GB files validate in O(1) memory.
      const lines = createInterface({ input: createReadStream(args.path, "utf8"), crlfDelay: Infinity });
      return validateJsonlLines(lines);
    },
  });
}
