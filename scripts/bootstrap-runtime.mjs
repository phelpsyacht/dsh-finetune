#!/usr/bin/env node
/**
 * Bootstrap a dsh-finetune Python runtime environment on macOS / Linux /
 * Windows: create a venv (default `~/.dsh-finetune/venv`, the same location
 * `lib/python.js` auto-detects) and install the trainer's only dependencies
 * (torch + transformers + peft).
 *
 * Usage:
 *   node scripts/bootstrap-runtime.mjs                  CPU build, default venv
 *   node scripts/bootstrap-runtime.mjs --dir /opt/dsh-venv
 *   node scripts/bootstrap-runtime.mjs --cuda cu124     NVIDIA CUDA torch wheels
 *   node scripts/bootstrap-runtime.mjs --index-url <url>
 *   node scripts/bootstrap-runtime.mjs --no-deps        create the venv only
 *
 * After success: when the default venv location is used the plugin finds it
 * automatically (zero config). Otherwise set plugin config `pythonPath` to the
 * printed interpreter path.
 */
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const IS_WIN = process.platform === "win32";
const PY_CHECK =
  "import sys; import torch, transformers, peft; " +
  "print(f\"{sys.version.split()[0]} torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__}\")";

function printUsage() {
  console.log(
    [
      "Bootstrap a dsh-finetune Python runtime (torch + transformers + peft).",
      "",
      "Usage: node scripts/bootstrap-runtime.mjs [options]",
      "  --dir <path>       venv directory (default ~/.dsh-finetune/venv)",
      "  --cuda <tag>       install CUDA torch, e.g. cu124 (default: CPU build)",
      "  --index-url <url>  torch install index override",
      "  --no-deps          only create the venv, skip dependency install",
      "  -h, --help         show this help",
      "",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const opts = { dir: join(homedir(), ".dsh-finetune", "venv"), indexUrl: null, installDeps: true, help: false };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "-h" || arg === "--help") opts.help = true;
    else if (arg === "--dir") opts.dir = argv[++i];
    else if (arg === "--cuda") opts.indexUrl = `https://download.pytorch.org/whl/${argv[++i] || "cu124"}`;
    else if (arg === "--index-url") opts.indexUrl = argv[++i];
    else if (arg === "--no-deps") opts.installDeps = false;
    else {
      console.error(`[bootstrap] unknown option: ${arg}`);
      opts.help = true;
    }
  }
  return opts;
}

function run(cmd, args, { allowFail = false, label = "" } = {}) {
  console.log(`[bootstrap] ${label || `${cmd} ${args.join(" ")}`}`);
  const res = spawnSync(cmd, args, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
  const out = `${res.stdout ?? ""}${res.stderr ?? ""}`.trim();
  if (!allowFail && res.status !== 0) {
    console.error(`[bootstrap] command failed (exit ${res.status}):\n${out}`);
    process.exit(1);
  }
  return { code: res.status, out };
}

/**
 * Find a base interpreter. `prefix` holds launcher flags that must precede
 * every `-m`/`-c` invocation (e.g. `py -3` on Windows).
 */
function findBaseInterpreter() {
  const versionProbe = "import sys; print(sys.version.split()[0])";
  const probes = IS_WIN
    ? [
        ["py", ["-3"]],
        ["python", []],
        ["python3", []],
      ]
    : [
        ["python3", []],
        ["python", []],
      ];
  for (const [cmd, prefix] of probes) {
    const res = spawnSync(cmd, [...prefix, "-c", versionProbe], { encoding: "utf8" });
    if (res.status === 0) return { cmd, prefix, version: res.stdout.trim() };
  }
  return null;
}

/** Paths of the interpreter inside a venv (platform-dependent). */
function venvInterpreter(dir) {
  return IS_WIN ? join(dir, "Scripts", "python.exe") : join(dir, "bin", "python");
}

function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help) {
    printUsage();
    return;
  }
  if (!opts.dir) {
    console.error("[bootstrap] --dir requires a path");
    process.exit(1);
  }

  const base = findBaseInterpreter();
  if (!base) {
    console.error("[bootstrap] no base Python found; install Python 3.10+ first (python.org or your package manager).");
    process.exit(1);
  }
  console.log(`[bootstrap] base interpreter: ${base.cmd} (Python ${base.version})`);

  const target = venvInterpreter(opts.dir);
  if (!existsSync(target)) {
    const venvArgs = [...base.prefix, "-m", "venv", opts.dir];
    run(base.cmd, venvArgs, { label: `create venv at ${opts.dir}` });
  } else {
    console.log(`[bootstrap] venv already exists: ${opts.dir}`);
  }

  if (!opts.installDeps) {
    console.log(`[bootstrap] --no-deps: venv ready at ${target}`);
    return;
  }

  run(target, ["-m", "pip", "install", "--upgrade", "pip"], { label: "upgrade pip" });

  // torch first from its own index (CPU by default on all platforms; CUDA via
  // --cuda/--index-url), then the PyPI-only packages.
  const cpuIndex = "https://download.pytorch.org/whl/cpu";
  if (opts.indexUrl) {
    run(target, ["-m", "pip", "install", "--index-url", opts.indexUrl, "torch"], { label: `install torch (${opts.indexUrl})` });
  } else if (IS_WIN || process.platform === "linux") {
    run(target, ["-m", "pip", "install", "--index-url", cpuIndex, "torch"], { label: `install torch (CPU, ${cpuIndex})` });
  } else {
    run(target, ["-m", "pip", "install", "torch"], { label: "install torch (PyPI)" });
  }
  run(target, ["-m", "pip", "install", "transformers", "peft"], { label: "install transformers + peft" });

  const check = run(target, ["-c", PY_CHECK], { label: "verify runtime", allowFail: true });
  if (check.code === 0) {
    console.log(`[bootstrap] OK: ${check.out}`);
  } else {
    console.error(`[bootstrap] runtime check failed:\n${check.out}`);
    process.exit(1);
  }

  const isDefault = opts.dir === join(homedir(), ".dsh-finetune", "venv");
  console.log("");
  console.log(isDefault
    ? "[bootstrap] default venv location used -> the plugin auto-detects it, no config needed."
    : `[bootstrap] set plugin config pythonPath to: ${target}`);
  console.log(`[bootstrap] done.`);
}

main();
