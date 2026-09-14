/**
 * Resolve the background-jobs registry (`@deepseek-ai/dsh-jobs-local`, composed
 * by `@deepseek-ai/dsh-base`) for one tool call.
 *
 * `jobs` is deliberately NOT part of the plugin's `inject` list. Cordis treats
 * every injected name as required: while one is missing the fiber never leaves
 * INACTIVE, `apply` never runs, and the failure is silent — a composition with
 * no jobs registry would lose all 8 finetune tools instead of just the
 * background-watch behaviour. Reading the service per call keeps every tool
 * registered, and lets each job-aware tool degrade on its own:
 * `finetune_job_watch` reports the missing registry, `finetune_train` falls
 * back to a detached process.
 *
 * `ctx.get(name)` is Cordis' "read a service without the inject requirement"
 * accessor — the same pattern `@deepseek-ai/dsh-acp` uses for `llm` and
 * `attachments`. Stub contexts (tests, embedders) expose the registry as a
 * plain property instead, so fall back to that when no accessor exists.
 *
 * @param ctx - the Cordis context (or a stub) the tool executes against.
 * @returns the registry, or `undefined` when none is composed.
 */
export function resolveJobs(ctx) {
  if (typeof ctx?.get === "function") return ctx.get("jobs");
  return ctx?.jobs;
}
