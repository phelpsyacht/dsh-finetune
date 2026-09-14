/**
 * dsh-finetune — LLM fine-tuning tools for the DeepSeek Harness.
 *
 * Plugin shape (named exports preserve loader injection metadata):
 * - `name`   plugin id used by Loader entries
 * - `inject` required Cordis services (`tools` from dsh-tools)
 * - `Config` schemastery schema for consumer config (patch-row `config:`)
 * - `apply`  registration; receives the validated config
 *
 * Design notes:
 * - Credentials never appear in tool arguments or the transcript: the API key
 *   is resolved from an environment variable named by config (`apiKeyEnv`).
 * - The provider is an adapter over an OpenAI-compatible fine-tuning REST
 *   surface (`/files`, `/fine_tuning/jobs`). Swap `lib/provider.js` for a
 *   different backend; the tools stay unchanged.
 * - `finetune_train` runs the bundled Python trainer either on this host
 *   (`transport: "local"`) or on a remote GPU box over ssh
 *   (`transport: "ssh"`, see `lib/remote.js`). Remote runs upload the dataset
 *   + config + trainer, launch detached on the target, and poll remotely.
 * - `finetune_job_create` / `finetune_job_cancel` are side-effectful, costly,
 *   and hard to reverse: they declare `isConcurrencySafe: false`. Deployments
 *   should gate them with an ask policy via permission presets
 *   (`tools/pre-execute`).
 * - `finetune_dataset_clean` applies a configurable cleaning pass (dedupe,
 *   length/ratio filters, text cleanup, assistant-ending enforcement) and
 *   writes a new file; the source dataset is never modified. `finetune_train`
 *   can run the same pass inline with its `clean` argument.
 */
import z from "@deepseek-ai/schemastery";
import { homedir } from "node:os";
import { join } from "node:path";
import { FinetuneProvider } from "./lib/provider.js";
import { datasetValidateTool } from "./lib/tools/dataset-validate.js";
import { datasetCleanTool } from "./lib/tools/dataset-clean.js";
import { jobCreateTool } from "./lib/tools/job-create.js";
import { jobStatusTool } from "./lib/tools/job-status.js";
import { jobListTool } from "./lib/tools/job-list.js";
import { jobCancelTool } from "./lib/tools/job-cancel.js";
import { jobWatchTool } from "./lib/tools/job-watch.js";
import { trainTool } from "./lib/tools/train.js";

const name = "finetune";
// `tools` is the only required service (the tool registry from dsh-tools, which
// every finetune tool registers into).
//
// `jobs` (the background-jobs registry from @deepseek-ai/dsh-jobs-local,
// composed by @deepseek-ai/dsh-base) is deliberately NOT injected here: Cordis
// treats every injected name as required, and while one is missing `apply`
// never runs — so a composition without a jobs registry would silently lose all
// 8 tools, not just background watching. `lib/tools/train.js` and
// `lib/tools/job-watch.js` resolve it lazily per call through
// `resolveJobs(ctx)` (lib/jobs.js) instead.
const inject = ["tools"];

const DEFAULTS = {
  baseURL: "https://api.deepseek.com",
  apiKeyEnv: "DEEPSEEK_API_KEY",
  defaultBaseModel: "deepseek-chat",
  pollIntervalMs: 60000,
  requestTimeoutMs: 120000,
  // Local (Python) training defaults. `pythonPath` empty = auto-detect an
  // interpreter with torch+transformers+peft installed.
  pythonPath: "",
  defaultLocalModel: "Qwen/Qwen3-0.6B",
  // Weight dtype for local training: "auto" is device-aware (bf16 on CUDA,
  // float32 on CPU — bf16 GEMM is pathologically slow on CPUs without native
  // bf16 support); "float32" / "bf16" force one. Overridable per call.
  dtype: "auto",
  trainingRoot: join(homedir(), ".dsh-finetune"),
  // SSH (remote-box) training defaults. `transport` selects the finetune_train
  // backend; "ssh" targets remote.host and uploads the dataset before launch.
  transport: "local",
  remote: {
    host: "",
    user: "",
    port: 22,
    keyPath: "",
    dir: "",
    python: "",
    strictHostKey: false,
  },
};

const Config = z.object({
  baseURL: z.string().description("Remote provider base URL."),
  apiKeyEnv: z.string().description("Environment variable holding the remote provider API key."),
  defaultBaseModel: z.string().description("Remote base model used when a create call names none."),
  pollIntervalMs: z.number().description("Default poll interval for job watching, in milliseconds."),
  requestTimeoutMs: z.number().description("Per-request HTTP timeout, in milliseconds."),
  pythonPath: z.string().description("Python interpreter with torch+transformers+peft; empty auto-detects."),
  defaultLocalModel: z.string().description("Default local base model/dir for finetune_train (HF id or path)."),
  dtype: z
    .string()
    .description('Weight dtype for finetune_train: "auto" (device-aware: bf16 on CUDA, float32 on CPU), "float32" or "bf16".'),
  trainingRoot: z.string().description("Directory under which runs/ are created for local training."),
  transport: z
    .string()
    .default("local")
    .description('finetune_train backend: "local" (spawn on this host) or "ssh" (run on remote.host).'),
  remote: z
    .object({
      host: z.string().description("SSH host of the training (GPU) box."),
      user: z.string().description("SSH username."),
      port: z.number().default(22).description("SSH port."),
      keyPath: z.string().default("").description("Path to the SSH private key; empty = ssh-agent / default ~/.ssh keys."),
      dir: z.string().default("").description('Remote working root under which runs/ is created; empty = "$HOME/.dsh-finetune".'),
      python: z.string().default("").description("Remote interpreter with torch+transformers+peft; empty = python3."),
      strictHostKey: z.boolean().default(false).description("Verify the host key strictly; false auto-accepts new hosts."),
    })
    .description('SSH backend settings used when transport is "ssh".'),
});

function apply(ctx, config) {
  const cfg = {
    ...DEFAULTS,
    ...config,
    remote: { ...DEFAULTS.remote, ...(config.remote ?? {}) },
  };
  const provider = new FinetuneProvider({
    baseURL: cfg.baseURL,
    apiKeyEnv: cfg.apiKeyEnv,
    requestTimeoutMs: cfg.requestTimeoutMs,
  });
  const env = { ctx, provider, config: cfg };
  for (const factory of [
    datasetValidateTool,
    datasetCleanTool,
    jobCreateTool,
    jobStatusTool,
    jobListTool,
    jobCancelTool,
    jobWatchTool,
    trainTool,
  ]) {
    ctx.tools.register(factory(env));
  }
}

export { name, inject, Config, apply };
