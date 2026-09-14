import { spawn } from "node:child_process";
import { openSync, writeSync, closeSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, basename } from "node:path";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { defineTool } from "@deepseek-ai/dsh-tools";
import { resolveJobs } from "../jobs.js";
import { validateJsonlDataset } from "../dataset.js";
import { cleanJsonlDataset, CLEAN_DEFAULTS } from "../clean.js";
import { resolveTrainPython } from "../python.js";
import { createSshTrainRunner, remoteExec, remoteHomeDir } from "../remote.js";

const TRAIN_PY = join(dirname(fileURLToPath(import.meta.url)), "..", "trainer", "train.py");
const SUPPORTED_METHODS = ["lora"];

const RESULT_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    runId: { type: "string" },
    watchJobId: { type: "string" },
    pid: { type: "string" },
    python: { type: "string" },
    pythonInfo: { type: "string" },
    outputDir: { type: "string" },
    configPath: { type: "string" },
    logPath: { type: "string" },
    method: { type: "string" },
    model: { type: "string" },
    dtype: { type: "string" },
    samples: { type: "integer" },
    nextStep: { type: "string" },
  },
};

function slugify(value) {
  const cleaned = String(value).toLowerCase().replace(/[^a-z0-9-_]+/g, "-").replace(/^-+|-+$/g, "");
  return cleaned || "run";
}

function makeRunId(args, datasetName) {
  return `${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}-${slugify(args.name ?? basename(datasetName, ".jsonl"))}`;
}

/** Merge user hyperparameter args into the trainer config JSON. */
function buildTrainConfig(args, overrides) {
  return {
    model_path: overrides.modelPath,
    data_path: overrides.dataPath,
    output_dir: overrides.outputDir,
    device: overrides.device,
    dtype: overrides.dtype,
    ...(args.lora_r !== undefined && { lora_r: args.lora_r }),
    ...(args.lora_alpha !== undefined && { lora_alpha: args.lora_alpha }),
    ...(args.learning_rate !== undefined && { learning_rate: args.learning_rate }),
    ...(args.epochs !== undefined && { num_epochs: args.epochs }),
    ...(args.max_seq_len !== undefined && { max_seq_len: args.max_seq_len }),
  };
}

/** Short prefix for `nextStep` when an inline cleaning pass ran. */
function cleanSummary(cleanResult) {
  return cleanResult
    ? `cleaned dataset: kept ${cleanResult.keptLines}/${cleanResult.inputLines} lines (removed ${cleanResult.removedLines}); `
    : "";
}

/**
 * Spawn the bundled trainer once, tee stdout+stderr to a log file, buffer the
 * text for job readOutput, and settle `done` when the process exits.
 */
function spawnTrainer(python, configPath, logPath) {
  const lines = [];
  let cursor = 0;
  const logFd = openSync(logPath, "a");

  const child = spawn(python, [TRAIN_PY, "--config", configPath], {
    env: { ...process.env, HF_HUB_OFFLINE: "1" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let logClosed = false;
  const closeLog = () => {
    if (logClosed) return;
    logClosed = true;
    try {
      closeSync(logFd);
    } catch {
      // already closed
    }
  };

  const onData = (chunk) => {
    const text = chunk.toString();
    lines.push(text);
    const joined = lines.join("\n");
    if (joined.length > 200_000) lines.shift();
    writeSync(logFd, text);
  };
  child.stdout.on("data", onData);
  child.stderr.on("data", onData);

  const done = new Promise((resolve) => {
    child.on("error", (error) => {
      closeLog();
      resolve({ status: "failed", detail: `spawn failed: ${error.message}`, output: lines.join("\n") });
    });
    child.on("close", (code) => {
      closeLog();
      resolve({
        status: code === 0 ? "completed" : "failed",
        detail: code === 0 ? "finished (exit 0)" : `exit code ${code} — see the log for the traceback`,
        output: lines.join("\n"),
      });
    });
  });

  return {
    child,
    done,
    kill: () => {
      if (child.exitCode !== null || child.signalCode !== null) return;
      child.kill("SIGTERM");
      const timer = setTimeout(() => child.kill("SIGKILL"), 10000);
      child.once("close", () => clearTimeout(timer));
    },
    readOutput: () => {
      const out = lines.slice(cursor).join("\n");
      cursor = lines.length;
      return out;
    },
  };
}

export function trainTool({ ctx, config }) {
  const transport = config.transport === "ssh" ? "ssh" : "local";

  /** Local backend: spawn the trainer on this host (original behaviour). */
  async function runLocal(args, exec, { dataset, cleanResult, dataPath }) {
    const runtime = resolveTrainPython(config);
    if (!runtime.python) throw new Error(`finetune: ${runtime.error}`);

    const runId = makeRunId(args, args.dataset_path);
    const outputDir = join(config.trainingRoot, "runs", runId);
    const logPath = join(outputDir, "train.log");
    await mkdir(outputDir, { recursive: true });

    if (cleanResult) {
      dataPath = join(outputDir, "cleaned.jsonl");
      await writeFile(dataPath, cleanResult.text);
    }

    const trainConfig = buildTrainConfig(args, {
      modelPath: args.model ?? config.defaultLocalModel,
      dataPath,
      outputDir,
      device: args.device ?? "cpu",
      dtype: args.dtype ?? config.dtype ?? "auto",
    });
    const configPath = join(outputDir, "config.json");
    await writeFile(configPath, JSON.stringify(trainConfig, null, 2));

    const common = {
      runId,
      outputDir,
      configPath,
      logPath,
      method: args.method ?? "lora",
      model: trainConfig.model_path,
      dtype: trainConfig.dtype,
      samples: cleanResult ? cleanResult.keptLines : dataset.lines,
      python: runtime.python,
      pythonInfo: runtime.info,
    };

    const jobs = resolveJobs(ctx);
    if (!jobs) {
      const runner = spawnTrainer(runtime.python, configPath, logPath);
      runner.child.unref();
      return {
        ...common,
        pid: String(runner.child.pid ?? ""),
        watchJobId: "",
        nextStep: `${cleanSummary(cleanResult)}no jobs registry composed; training detached (pid ${runner.child.pid}). Poll ${logPath} with fs/bash tools; adapter lands in ${outputDir}`,
      };
    }

    let runner;
    const watchJobId = jobs.start({
      kind: "finetune",
      label: `finetune ${runId}`,
      owner: exec.agent,
      run() {
        runner = spawnTrainer(runtime.python, configPath, logPath);
        return {
          cancel: (reason) => runner.kill(),
          done: runner.done,
          readOutput: runner.readOutput,
        };
      },
    });
    const onToolAbort = () => runner?.kill();
    exec.signal?.addEventListener("abort", onToolAbort, { once: true });

    return {
      ...common,
      watchJobId: String(watchJobId),
      pid: "",
      nextStep: `${cleanSummary(cleanResult)}training launched; run job_output ${watchJobId} (or read ${logPath}) for progress; adapter will be saved to ${outputDir}`,
    };
  }

  /** SSH backend: upload the dataset + trainer to remote.host, run there. */
  async function runSsh(args, exec, { dataset, cleanResult, dataPath }) {
    const remote = config.remote ?? {};
    if (!remote.host || !remote.user) {
      throw new Error(
        "finetune: transport is 'ssh' but remote.host / remote.user are not set in the plugin config (config.remote)",
      );
    }

    // Fail fast on unreachable host / bad key before any upload happens.
    const ping = await remoteExec(remote, "true", { timeoutMs: 20000 });
    if (ping.code !== 0) {
      throw new Error(
        `finetune: cannot reach ssh target ${remote.user}@${remote.host}: ${(ping.stderr || ping.stdout).trim().slice(0, 400)}`,
      );
    }

    const python = remote.python || "python3";
    // Absolute remote base dir: honour config.remote.dir, else $HOME/.dsh-finetune.
    let base = remote.dir;
    if (!base) {
      const home = await remoteHomeDir(remote);
      base = `${home}/.dsh-finetune`;
    }
    const hostLabel = `${remote.user}@${remote.host}`;

    const runId = makeRunId(args, args.dataset_path);
    const localMetaDir = join(config.trainingRoot, "runs", runId);
    const runDir = `${base}/runs/${runId}`;
    await mkdir(localMetaDir, { recursive: true });

    if (cleanResult) {
      dataPath = join(localMetaDir, "cleaned.jsonl");
      await writeFile(dataPath, cleanResult.text);
    }

    // data_path/output_dir already point at the REMOTE filesystem; the config
    // is uploaded next to the dataset and the bundled trainer.
    const trainConfig = buildTrainConfig(args, {
      modelPath: args.model ?? config.defaultLocalModel,
      dataPath: `${runDir}/${basename(dataPath)}`,
      outputDir: runDir,
      device: args.device ?? "auto",
      dtype: args.dtype ?? config.dtype ?? "auto",
    });
    const configPath = join(localMetaDir, "config.json");
    await writeFile(configPath, JSON.stringify(trainConfig, null, 2));

    const common = {
      runId,
      outputDir: runDir,
      configPath,
      logPath: `${runDir}/train.log`,
      method: args.method ?? "lora",
      model: trainConfig.model_path,
      dtype: trainConfig.dtype,
      samples: cleanResult ? cleanResult.keptLines : dataset.lines,
      python,
      pythonInfo: `ssh ${hostLabel}: ${python}`,
    };

    const runner = createSshTrainRunner({
      remote,
      runId,
      python,
      localTrainPy: TRAIN_PY,
      localConfigPath: configPath,
      datasetPath: dataPath,
      runDir,
      pollIntervalMs: config.pollIntervalMs,
    });

    const jobs = resolveJobs(ctx);
    if (!jobs) {
      return {
        ...common,
        pid: runner.pid,
        watchJobId: "",
        nextStep: `${cleanSummary(cleanResult)}no jobs registry composed; training detached on ${hostLabel}. Poll ${runDir}/train.log over ssh (or read ${configPath} locally); adapter lands in ${runDir}`,
      };
    }

    const watchJobId = jobs.start({
      kind: "finetune",
      label: `finetune ${runId}`,
      owner: exec.agent,
      run() {
        return {
          cancel: (reason) => runner.kill(),
          done: runner.done,
          readOutput: runner.readOutput,
        };
      },
    });
    const onToolAbort = () => runner.kill();
    exec.signal?.addEventListener("abort", onToolAbort, { once: true });

    return {
      ...common,
      watchJobId: String(watchJobId),
      pid: runner.pid,
      nextStep: `${cleanSummary(cleanResult)}training launched on ${hostLabel}; run job_output ${watchJobId} for progress (or ssh tail ${runDir}/train.log); adapter will be saved to ${runDir}`,
    };
  }

  return defineTool({
    name: "finetune_train",
    description:
      "Run LoRA SFT fine-tuning on a JSONL dataset using the bundled trainer (torch + transformers + peft only — no external training framework). " +
      `Backend is selected by plugin config: transport "local" spawns the trainer on this host; transport "ssh" uploads the dataset + trainer to config.remote.host (a remote GPU box holding the model) and runs there. ` +
      "Pre-flights the dataset, writes a run config, then launches training as a background job (kind `finetune`) whose output is streamed to a log file. " +
      `Supported methods: ${SUPPORTED_METHODS.join(", ")}. Monitor with job_output / job_list; the session is notified on completion. ` +
      "Training is slow — confirm dataset size and expected runtime with the user first. " +
      "Pass clean=true to run the built-in data cleaner (dedupe, length/ratio filters, text cleanup) on a copy before training.",
    parameters: {
      dataset_path: { type: "string", required: true, description: "Absolute local path to the .jsonl dataset (chat / prompt-completion / alpaca / sharegpt); uploaded to the ssh target when transport is ssh" },
      clean: {
        type: "boolean",
        description:
          "Run the built-in data cleaner on the dataset before training (dedupe, min/max length, repeat/meaningful-ratio filters, control-char/HTML/URL removal, assistant-ending enforcement, same-role merge). " +
          "Trains on a cleaned copy written next to the run; the original file is never modified (default false)",
      },
      name: { type: "string", description: "Short run name used for the output directory and job label" },
      model: { type: "string", description: "Base model for ssh: a path or HF id reachable on the remote box (offline HF cache); for local: local dir or HF id in the local offline cache (default from plugin config)" },
      method: { type: "string", description: `Fine-tuning method (default lora; currently supported: ${SUPPORTED_METHODS.join(", ")})` },
      lora_r: { type: "integer", description: "LoRA rank (default 8)" },
      lora_alpha: { type: "integer", description: "LoRA alpha (default 16)" },
      learning_rate: { type: "number", description: "Learning rate (default 2e-4)" },
      epochs: { type: "number", description: "Training epochs (default 3)" },
      max_seq_len: { type: "integer", description: "Token context window per sample (default 1024)" },
      device: { type: "string", description: "cpu, auto (cuda when available) or cuda; default cpu locally, auto over ssh" },
      dtype: {
        type: "string",
        description:
          "Model weight dtype: auto (default), float32 or bf16. auto is device-aware — bf16 on CUDA, float32 on CPU, " +
          "because bf16 GEMM is pathologically slow on CPUs without native bf16 support (e.g. AVX2-only Intel). " +
          "Pass bf16 explicitly to force bf16 on a CPU.",
      },
    },
    output: {
      schema: RESULT_SCHEMA,
      render: (_args, value) => [
        {
          type: "text",
          text: `finetune run ${value.runId}: ${value.method} on ${value.model}, ${value.samples} samples, dtype ${value.dtype}` +
            ` | python ${value.pythonInfo} | logs: ${value.logPath}` +
            (value.watchJobId ? ` | background job ${value.watchJobId}` : ` | pid ${value.pid}`),
        },
      ],
    },
    isConcurrencySafe: () => false,
    timeoutMs: 60000,
    async execute(args, exec) {
      if (args.method && !SUPPORTED_METHODS.includes(args.method)) {
        throw new Error(`finetune: method "${args.method}" not supported yet; available: ${SUPPORTED_METHODS.join(", ")}`);
      }

      const text = await readFile(args.dataset_path, "utf8");
      const dataset = validateJsonlDataset(text);
      if (!dataset.valid) {
        const first = dataset.errors[0]
          ? ` (first error at line ${dataset.errors[0].line}: ${dataset.errors[0].message})`
          : " (file is empty)";
        throw new Error(`finetune: refusing to train on an invalid dataset${first}`);
      }

      let cleanResult = null;
      let dataPath = args.dataset_path;
      if (args.clean) {
        cleanResult = cleanJsonlDataset(text, CLEAN_DEFAULTS);
        if (cleanResult.keptLines === 0) {
          throw new Error("finetune: cleaning removed every record; refusing to train on an empty dataset");
        }
      }

      const state = { dataset, cleanResult, dataPath };
      return transport === "ssh" ? runSsh(args, exec, state) : runLocal(args, exec, state);
    },
  });
}
