import { readFile } from "node:fs/promises";
import { defineTool } from "@deepseek-ai/dsh-tools";
import { validateJsonlDataset } from "../dataset.js";

const RESULT_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    jobId: { type: "string" },
    fileId: { type: "string" },
    model: { type: "string" },
    status: { type: "string" },
    nextStep: { type: "string" },
  },
};

export function jobCreateTool({ provider, config }) {
  return defineTool({
    name: "finetune_job_create",
    description:
      "Upload a validated JSONL dataset and start a fine-tuning job on the configured provider. Costly, slow, and hard to reverse: run finetune_dataset_validate first, summarize the dataset (size, format, sample) and the target base model to the user, and get explicit confirmation before calling. Poll progress with finetune_job_status or arm finetune_job_watch.",
    parameters: {
      dataset_path: { type: "string", required: true, description: "Absolute path to the .jsonl training file" },
      base_model: {
        type: "string",
        description: `Base model to fine-tune; defaults to "${config.defaultBaseModel}" from plugin config`,
      },
      suffix: { type: "string", description: "Name suffix for the resulting fine-tuned model (letters, digits, - and _)" },
      epochs: { type: "number", description: "Number of training epochs; provider default applies when omitted" },
    },
    output: {
      schema: RESULT_SCHEMA,
      render: (_args, value) => [
        { type: "text", text: `fine-tune job ${value.jobId} created (${value.status}) on ${value.model}; ${value.nextStep}` },
      ],
    },
    isConcurrencySafe: () => false,
    timeoutMs: 600000,
    async execute(args, exec) {
      const dataset = validateJsonlDataset(await readFile(args.dataset_path, "utf8"));
      if (!dataset.valid) {
        const first = dataset.errors[0]
          ? ` (first error at line ${dataset.errors[0].line}: ${dataset.errors[0].message})`
          : "";
        throw new Error(`finetune: refusing to upload an invalid dataset${first}; fix the file and re-run finetune_dataset_validate`);
      }
      const file = await provider.uploadDataset(args.dataset_path, exec.signal);
      const job = await provider.createJob(
        {
          trainingFile: file.id,
          model: args.base_model ?? config.defaultBaseModel,
          ...(args.suffix !== undefined && { suffix: args.suffix }),
          ...(args.epochs !== undefined && { hyperparameters: { nEpochs: args.epochs } }),
        },
        exec.signal,
      );
      return {
        jobId: job.jobId,
        fileId: file.id,
        model: job.model,
        status: job.status,
        nextStep: "poll with finetune_job_status, or arm finetune_job_watch for a background completion notice",
      };
    },
  });
}
