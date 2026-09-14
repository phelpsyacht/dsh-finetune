/**
 * End-to-end real-training verification for dsh-finetune.
 *
 * Drives the plugin's own tools (dataset_validate / dataset_clean / finetune_train)
 * with the REAL local environment: no fake python, no stubbed jobs — the jobs
 * registry actually runs the spawned trainer and we await completion.
 *
 * Usage:
 *   DATASET_PATH=/abs/path/to/data.jsonl node tests/e2e-real-train.mjs
 *
 * Config under test (mirrors the plugin defaults for this machine):
 *   pythonPath        = ~/CodeBuddy/.venv/bin/python  (torch 2.2.2 / transformers 4.57.6 / peft 0.20.0)
 *   defaultLocalModel = Qwen/Qwen3-0.6B               (offline HF cache)
 *   trainingRoot      = ~/.dsh-finetune
 */
import { trainTool } from "../lib/tools/train.js";
import { datasetValidateTool } from "../lib/tools/dataset-validate.js";
import { datasetCleanTool } from "../lib/tools/dataset-clean.js";
import * as plugin from "../index.js";

const DATASET = process.env.DATASET_PATH;
if (!DATASET) {
  console.error("set DATASET_PATH to the .jsonl dataset to train on");
  process.exit(2);
}

// jobs registry that actually runs the job so the harness can await completion
const jobState = {};
const jobs = {
  start(spec) {
    jobState.spec = spec;
    jobState.job = spec.run();
    return 1; // watchJobId
  },
};

const config = {
  transport: "local",
  pythonPath: "/Users/phelps/CodeBuddy/.venv/bin/python",
  defaultLocalModel: "Qwen/Qwen3-0.6B",
  trainingRoot: "/Users/phelps/.dsh-finetune",
};

console.log("== plugin wiring ==");
console.log(JSON.stringify({ name: plugin.name, inject: plugin.inject, hasConfig: typeof plugin.Config === "function", hasApply: typeof plugin.apply === "function" }));

const validate = datasetValidateTool();
const clean = datasetCleanTool();
const train = trainTool({ ctx: { jobs }, config });

console.log("\n== finetune_dataset_validate ==");
const v = await validate.execute({ path: DATASET }, {});
console.log(JSON.stringify({ valid: v.valid, lines: v.lines, format: v.format, approxTokens: v.approxTokens, errors: v.errors, preview: v.preview?.slice(0, 160) }));

console.log("\n== finetune_dataset_clean (dry-run) ==");
const c = await clean.execute({ path: DATASET, dry_run: true }, {});
console.log(JSON.stringify({ inputLines: c.inputLines, keptLines: c.keptLines, removedLines: c.removedLines, byRule: c.byRule, examples: c.examples }));

console.log("\n== finetune_train (real run) ==");
const result = await train.execute(
  {
    dataset_path: DATASET,
    name: "demo-verify",
    epochs: 2,
    max_seq_len: 256,
    device: "cpu",
  },
  {},
);
console.log(JSON.stringify({ runId: result.runId, model: result.model, samples: result.samples, python: result.python, pythonInfo: result.pythonInfo, watchJobId: result.watchJobId, outputDir: result.outputDir, configPath: result.configPath, logPath: result.logPath, nextStep: result.nextStep }));

console.log("\n== awaiting trainer process ==");
const outcome = await jobState.job.done;
const tail = jobState.job.readOutput();
console.log(JSON.stringify({ status: outcome.status, detail: outcome.detail }));
console.log("--- trainer output tail ---");
console.log(tail.slice(-4000));
