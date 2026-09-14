import { defineTool } from "@deepseek-ai/dsh-tools";

const RESULT_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    jobId: { type: "string" },
    status: { type: "string" },
    detail: { type: "string" },
  },
};

export function jobCancelTool({ provider }) {
  return defineTool({
    name: "finetune_job_cancel",
    description:
      "Cancel a running fine-tuning job. Irreversible: already-spent training cost is not refunded. Confirm with the user before calling.",
    parameters: {
      jobId: { type: "string", required: true, description: "Fine-tuning job id to cancel" },
    },
    output: {
      schema: RESULT_SCHEMA,
      render: (_args, value) => [{ type: "text", text: `job ${value.jobId}: ${value.status}` }],
    },
    isConcurrencySafe: () => false,
    timeoutMs: 30000,
    async execute(args, exec) {
      const job = await provider.cancelJob(args.jobId, exec.signal);
      return { jobId: job.jobId, status: job.status, detail: job.detail };
    },
  });
}
