import { defineTool } from "@deepseek-ai/dsh-tools";
import { resolveJobs } from "../jobs.js";

const RESULT_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    jobId: { type: "string" },
    watchJobId: { type: "string" },
    pollIntervalMs: { type: "integer" },
  },
};

/** Abortable sleep: resolves early when `signal` fires. */
function sleep(ms, signal) {
  return new Promise((resolve) => {
    const finish = () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", finish);
      resolve();
    };
    const timer = setTimeout(finish, ms);
    signal?.addEventListener("abort", finish, { once: true });
  });
}

const MAX_CONSECUTIVE_ERRORS = 3;

export function jobWatchTool({ ctx, provider, config }) {
  return defineTool({
    name: "finetune_job_watch",
    description:
      "Watch a fine-tuning job in the background: starts a background job (kind `finetune`) that polls the provider until the training job reaches a terminal state, then reports in-session. Manage the watcher with the generic job_output / job_list / job_kill tools. Requires a composed jobs registry; if unavailable, poll with finetune_job_status instead.",
    parameters: {
      jobId: { type: "string", required: true, description: "Fine-tuning job id to watch" },
      pollIntervalMs: {
        type: "integer",
        description: `Poll interval in milliseconds (default ${config.pollIntervalMs} from plugin config)`,
      },
    },
    output: {
      schema: RESULT_SCHEMA,
      render: (_args, value) => [
        { type: "text", text: `watching job ${value.jobId} as background job ${value.watchJobId} every ${value.pollIntervalMs}ms` },
      ],
    },
    isConcurrencySafe: () => false,
    timeoutMs: 15000,
    async execute(args, exec) {
      const jobs = resolveJobs(ctx);
      if (!jobs) {
        throw new Error(
          "finetune: no background-jobs registry is composed (load dsh-jobs-local / dsh-tool-jobs); poll with finetune_job_status instead",
        );
      }
      const intervalMs = args.pollIntervalMs ?? config.pollIntervalMs;
      const controller = new AbortController();
      const lines = [];
      let cursor = 0;
      let lastStatus = null;

      let watchJobId;
      try {
        watchJobId = jobs.start({
          kind: "finetune",
          label: `finetune ${args.jobId}`,
          owner: exec.agent,
          run() {
            const done = (async () => {
              let consecutiveErrors = 0;
              while (!controller.signal.aborted) {
                let job;
                try {
                  job = await provider.getJob(args.jobId, controller.signal);
                  consecutiveErrors = 0;
                } catch (error) {
                  if (controller.signal.aborted) break;
                  consecutiveErrors += 1;
                  lines.push(`[${new Date().toISOString()}] poll error: ${error.message}`);
                  if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
                    return { status: "failed", detail: `polling ${args.jobId} failed ${MAX_CONSECUTIVE_ERRORS} times in a row`, output: lines.join("\n") };
                  }
                  await sleep(intervalMs, controller.signal);
                  continue;
                }
                if (job.status !== lastStatus) {
                  lastStatus = job.status;
                  lines.push(`[${new Date().toISOString()}] status: ${job.status}`);
                }
                if (job.terminal) {
                  lines.push(`final: ${JSON.stringify(job)}`);
                  return {
                    status: job.status === "succeeded" ? "completed" : "failed",
                    detail: `finetune ${args.jobId}: ${job.status}${job.fineTunedModel ? ` → ${job.fineTunedModel}` : ""}`,
                    output: lines.join("\n"),
                  };
                }
                await sleep(intervalMs, controller.signal);
              }
              return { status: "killed", detail: `watch on ${args.jobId} cancelled` };
            })();
            return {
              cancel: (reason) => controller.abort(new Error(reason ?? "cancelled")),
              done,
              readOutput: () => {
                const out = lines.slice(cursor).join("\n");
                cursor = lines.length;
                return out;
              },
            };
          },
        });
      } catch (error) {
        throw new Error(`finetune: failed to start background watch: ${error.message}`);
      }
      return { jobId: args.jobId, watchJobId: String(watchJobId), pollIntervalMs: intervalMs };
    },
  });
}
