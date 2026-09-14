import { defineTool } from "@deepseek-ai/dsh-tools";

const RESULT_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    jobId: { type: "string" },
    status: { type: "string" },
    terminal: { type: "boolean" },
    model: { type: "string" },
    fineTunedModel: { type: "string" },
    createdAt: { type: "string" },
    detail: { type: "string" },
  },
};

export function jobStatusTool({ provider }) {
  return defineTool({
    name: "finetune_job_status",
    description:
      "Fetch the current status of one fine-tuning job. Read-only; safe to call repeatedly. When the job succeeds the fine-tuned model name is returned in fineTunedModel.",
    parameters: {
      jobId: { type: "string", required: true, description: "Fine-tuning job id returned by finetune_job_create" },
    },
    output: {
      schema: RESULT_SCHEMA,
      render: (_args, value) => [
        {
          type: "text",
          text: `job ${value.jobId}: ${value.status}${value.fineTunedModel ? ` → ${value.fineTunedModel}` : ""}${
            value.detail ? ` (${value.detail})` : ""
          }`,
        },
      ],
    },
    isConcurrencySafe: () => true,
    timeoutMs: 30000,
    async execute(args, exec) {
      return provider.getJob(args.jobId, exec.signal);
    },
  });
}
