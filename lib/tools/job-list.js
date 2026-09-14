import { defineTool } from "@deepseek-ai/dsh-tools";

const JOB_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    jobId: { type: "string" },
    status: { type: "string" },
    terminal: { type: "boolean" },
    model: { type: "string" },
    fineTunedModel: { type: "string" },
    createdAt: { type: "string" },
  },
};

export function jobListTool({ provider }) {
  return defineTool({
    name: "finetune_job_list",
    description: "List recent fine-tuning jobs on the configured provider (most recent first). Read-only.",
    parameters: {
      limit: { type: "integer", description: "Maximum number of jobs to return (default 10)" },
    },
    output: {
      schema: { type: "array", items: JOB_SCHEMA },
      render: (_args, jobs) => [
        {
          type: "text",
          text:
            jobs.length === 0
              ? "no fine-tuning jobs"
              : jobs.map((job) => `${job.jobId}: ${job.status}${job.fineTunedModel ? ` → ${job.fineTunedModel}` : ""}`).join("\n"),
        },
      ],
    },
    isConcurrencySafe: () => true,
    timeoutMs: 30000,
    async execute(args, exec) {
      return provider.listJobs(args.limit ?? 10, exec.signal);
    },
  });
}
