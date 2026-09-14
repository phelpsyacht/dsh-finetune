import { test } from "node:test";
import assert from "node:assert/strict";
import { apply, inject } from "../index.js";

/** Boot the plugin against a stubbed ctx and collect registered tools. */
function collectTools(config = {}) {
  const tools = [];
  const ctx = {
    tools: { register: (tool) => tools.push(tool) },
    jobs: undefined,
  };
  apply(ctx, config);
  return tools;
}

test("apply registers all 8 tools under the finetune_ prefix", () => {
  const names = collectTools().map((tool) => tool.name).sort();
  assert.deepEqual(names, [
    "finetune_dataset_clean",
    "finetune_dataset_validate",
    "finetune_job_cancel",
    "finetune_job_create",
    "finetune_job_list",
    "finetune_job_status",
    "finetune_job_watch",
    "finetune_train",
  ]);
});

test("clean tool exposes every documented parameter", () => {
  const clean = collectTools().find((tool) => tool.name === "finetune_dataset_clean");
  for (const param of [
    "path",
    "out",
    "dry_run",
    "dedupe",
    "min_chars",
    "max_chars",
    "repeat_ratio",
    "min_meaningful_ratio",
    "require_assistant_end",
    "clean_text",
    "merge_same_role",
    "collapse_space",
  ]) {
    assert.ok(param in clean.parameters.properties, `missing parameter: ${param}`);
  }
  assert.ok(clean.parameters.required.includes("path"), "path should be required");
  assert.equal(typeof clean.execute, "function");
});

test("train tool exposes the clean parameter", () => {
  const train = collectTools().find((tool) => tool.name === "finetune_train");
  assert.ok("clean" in train.parameters.properties, "missing parameter: clean");
  assert.ok(train.parameters.required.includes("dataset_path"), "dataset_path should be required");
});

test("plugin accepts a config with ssh defaults and still boots", () => {
  const tools = collectTools({ transport: "ssh", remote: { host: "gpu", user: "root" } });
  assert.equal(tools.length, 8);
});

test("jobs is not a required inject", () => {
  // Cordis gates `apply` on every injected name, so declaring `jobs` there
  // would make a composition that loads no jobs registry lose all 8 tools
  // silently instead of just background watching. The job-aware tools resolve
  // it lazily per call (lib/jobs.js).
  assert.deepEqual(inject, ["tools"]);
});

test("apply registers all 8 tools when the ctx has no jobs property at all", () => {
  const tools = [];
  apply({ tools: { register: (tool) => tools.push(tool) } }, {});
  assert.equal(tools.length, 8);
  assert.ok(tools.some((tool) => tool.name === "finetune_job_watch"));
});
