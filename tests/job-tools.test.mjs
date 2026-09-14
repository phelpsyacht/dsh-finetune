import { test } from "node:test";
import assert from "node:assert/strict";
import { ToolArgsError } from "@deepseek-ai/dsh-tools";
import { FinetuneProvider } from "../lib/provider.js";
import { jobCreateTool } from "../lib/tools/job-create.js";
import { jobStatusTool } from "../lib/tools/job-status.js";
import { jobListTool } from "../lib/tools/job-list.js";
import { jobCancelTool } from "../lib/tools/job-cancel.js";
import { jobWatchTool } from "../lib/tools/job-watch.js";
import { resolveJobs } from "../lib/jobs.js";

// Provider that never talks to the network: the tests below only exercise the
// argument-validation layer and the fail-fast paths that precede any request.
const provider = new FinetuneProvider({ baseURL: "http://127.0.0.1:1", apiKeyEnv: "DSH_FINETUNE_TEST_NONE" });
const env = { ctx: { jobs: undefined }, provider, config: {} };
const tools = {
  create: jobCreateTool(env),
  status: jobStatusTool(env),
  list: jobListTool(env),
  cancel: jobCancelTool(env),
  watch: jobWatchTool(env),
};

test("job tools reject missing required arguments with ToolArgsError", async () => {
  await assert.rejects(() => tools.create.execute({}, {}), ToolArgsError);
  await assert.rejects(() => tools.status.execute({}, {}), ToolArgsError);
  await assert.rejects(() => tools.cancel.execute({}, {}), ToolArgsError);
  await assert.rejects(() => tools.watch.execute({}, {}), ToolArgsError);
});

test("job tools reject wrong argument types with ToolArgsError", async () => {
  await assert.rejects(() => tools.status.execute({ jobId: 123 }, {}), ToolArgsError);
  await assert.rejects(() => tools.watch.execute({ jobId: null }, {}), ToolArgsError);
});

test("job-create: a non-existent dataset fails at file read, before any network call", async () => {
  await assert.rejects(
    () => tools.create.execute({ dataset_path: "/nonexistent/definitely-missing.jsonl" }, {}),
    /ENOENT|no such file/,
  );
});

test("job-list: wrong limit type is rejected by the schema", async () => {
  await assert.rejects(() => tools.list.execute({ limit: true }, {}), ToolArgsError);
});

test("finetune tools declare the expected concurrency safety", () => {
  assert.equal(tools.create.isConcurrencySafe({ dataset_path: "/tmp/x.jsonl" }), false);
  assert.equal(tools.cancel.isConcurrencySafe({ jobId: "x" }), false);
  assert.equal(tools.status.isConcurrencySafe({ jobId: "x" }), true);
  // invalid args short-circuit to unsafe, even on read-only tools
  assert.equal(tools.status.isConcurrencySafe({}), false);
});

test("resolveJobs: reads the registry through ctx.get, with no inject declaration", () => {
  const registry = { start: () => 7 };
  assert.equal(resolveJobs({ get: (name) => (name === "jobs" ? registry : undefined) }), registry);
  // a ctx whose accessor finds nothing must not fall back to a throwing read
  assert.equal(resolveJobs({ get: () => undefined }), undefined);
  // stub contexts (tests, embedders) expose the registry as a plain property
  assert.equal(resolveJobs({ jobs: registry }), registry);
  assert.equal(resolveJobs({ jobs: undefined }), undefined);
  assert.equal(resolveJobs(undefined), undefined);
});

test("job-watch: reports a missing jobs registry at call time, not at load time", async () => {
  await assert.rejects(
    () => tools.watch.execute({ jobId: "ft-1" }, {}),
    /no background-jobs registry is composed/,
  );
});
