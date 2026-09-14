import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { trainTool } from "../lib/tools/train.js";
import { writeJsonl, testDir, VALID_CHAT } from "./helpers.mjs";

const FIXTURE_PYTHON = join(dirname(fileURLToPath(import.meta.url)), "fixtures", "fake-python.sh");

/**
 * A jobs registry stub that records the job spec but never runs it, so the
 * tests exercise the real harness code path (jobs.start -> watchJobId) without
 * spawning the bundled Python trainer.
 */
function makeJobs() {
  const registered = [];
  return {
    registered,
    start: (spec) => {
      registered.push(spec);
      return 42; // fake watchJobId
    },
  };
}

function makeTrain(extraConfig = {}) {
  const jobs = makeJobs();
  const train = trainTool({
    ctx: { jobs },
    config: {
      transport: "local",
      // fake-python.sh satisfies resolveTrainPython's probe without torch;
      // the jobs stub never spawns it, so no real training is triggered.
      pythonPath: FIXTURE_PYTHON,
      trainingRoot: join(testDir(), "runs"),
      defaultLocalModel: "Qwen/Qwen3-0.6B",
      ...extraConfig,
    },
  });
  return { train, jobs };
}

test("finetune_train: refuses an invalid dataset before anything else", async () => {
  const { train } = makeTrain();
  const path = writeJsonl("train-invalid.jsonl", ["not json"]);
  await assert.rejects(() => train.execute({ dataset_path: path }, {}), /refusing to train on an invalid dataset/);
});

test("finetune_train: clean=true refuses when cleaning removes every record", async () => {
  const { train } = makeTrain();
  const path = writeJsonl("train-all-removed.jsonl", [
    JSON.stringify({ messages: [{ role: "user", content: "ok" }, { role: "assistant", content: "y" }] }),
  ]);
  await assert.rejects(
    () => train.execute({ dataset_path: path, clean: true }, {}),
    /cleaning removed every record/,
  );
});

test("finetune_train: clean=true writes a cleaned copy, trains on it, and never touches the input", async () => {
  const { train, jobs } = makeTrain();
  const path = writeJsonl("train-clean.jsonl", [
    VALID_CHAT,
    VALID_CHAT, // duplicate — removed by cleaning
    JSON.stringify({ messages: [{ role: "user", content: "1加1等于几？" }, { role: "assistant", content: "等于2，因为1加1就是2。" }] }),
  ]);
  const before = readFileSync(path, "utf8");

  const result = await train.execute({ dataset_path: path, clean: true, name: "clean-run" }, {});
  assert.equal(result.samples, 2); // 3 lines minus the duplicate
  assert.equal(result.watchJobId, "42");
  assert.match(result.nextStep, /cleaned dataset: kept 2\/3 lines \(removed 1\)/);

  const cleanedPath = join(result.outputDir, "cleaned.jsonl");
  assert.ok(existsSync(cleanedPath), "cleaned copy missing");
  const cleanedRecords = readFileSync(cleanedPath, "utf8").trim().split("\n");
  assert.equal(cleanedRecords.length, 2);

  const config = JSON.parse(readFileSync(result.configPath, "utf8"));
  assert.equal(config.data_path, cleanedPath); // trainer reads the cleaned copy
  assert.equal(config.output_dir, result.outputDir);

  assert.equal(readFileSync(path, "utf8"), before); // input never modified
  assert.equal(jobs.registered.length, 1);
  assert.equal(jobs.registered[0].kind, "finetune");
});

test("finetune_train: clean=false keeps original behaviour (trains on the source file)", async () => {
  const { train } = makeTrain();
  const path = writeJsonl("train-default.jsonl", [VALID_CHAT]);
  const result = await train.execute({ dataset_path: path, name: "plain-run" }, {});
  assert.equal(result.samples, 1);
  assert.doesNotMatch(result.nextStep, /cleaned/);
  assert.equal(existsSync(join(result.outputDir, "cleaned.jsonl")), false);
  assert.equal(JSON.parse(readFileSync(result.configPath, "utf8")).data_path, path);
});

test("finetune_train: dtype defaults to auto and a call argument reaches the trainer config", async () => {
  const { train } = makeTrain();
  const path = writeJsonl("train-dtype.jsonl", [VALID_CHAT]);

  const auto = await train.execute({ dataset_path: path, name: "dtype-auto" }, {});
  assert.equal(auto.dtype, "auto");
  assert.equal(JSON.parse(readFileSync(auto.configPath, "utf8")).dtype, "auto");

  const forced = await train.execute({ dataset_path: path, name: "dtype-f32", dtype: "float32" }, {});
  assert.equal(forced.dtype, "float32");
  assert.equal(JSON.parse(readFileSync(forced.configPath, "utf8")).dtype, "float32");
});

test("finetune_train: plugin config dtype is the default and a call argument overrides it", async () => {
  const { train } = makeTrain({ dtype: "float32" });
  const path = writeJsonl("train-dtype-config.jsonl", [VALID_CHAT]);

  const fromConfig = await train.execute({ dataset_path: path, name: "dtype-cfg" }, {});
  assert.equal(fromConfig.dtype, "float32");

  const overridden = await train.execute({ dataset_path: path, name: "dtype-call", dtype: "bf16" }, {});
  assert.equal(overridden.dtype, "bf16");
  assert.equal(JSON.parse(readFileSync(overridden.configPath, "utf8")).dtype, "bf16");
});

test("finetune_train: rejects unsupported methods", async () => {
  const { train } = makeTrain();
  const path = writeJsonl("train-method.jsonl", [VALID_CHAT]);
  await assert.rejects(
    () => train.execute({ dataset_path: path, method: "qlora" }, {}),
    /method "qlora" not supported/,
  );
});

test("finetune_train: dataset_path is required", async () => {
  const { train } = makeTrain();
  await assert.rejects(() => train.execute({}, {}), /required/);
});
