/**
 * Locate a Python interpreter that can run the bundled trainer
 * (torch + transformers + peft present), with zero required external setup:
 * resolution order = plugin config `pythonPath` → env `DSH_FINETUNE_PYTHON` →
 * a well-known local venv → an interpreter on PATH.
 *
 * Cross-platform by design:
 *  - macOS / Linux (POSIX) venvs keep the interpreter at `<venv>/bin/python`
 *    and commonly expose `python3` / `python` on PATH.
 *  - Windows venvs keep it at `<venv>\Scripts\python.exe` and commonly expose
 *    `python` / `python3` or the `py -3` launcher.
 * Candidates that do not exist are skipped without spawning anything, so extra
 * entries are harmless on any platform.
 */
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const IS_WIN = process.platform === "win32";

const PY_CHECK =
  "import sys; import torch, transformers, peft; " +
  "print(f\"{sys.version.split()[0]} torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__}\")";

const PY_EXE = "import sys; print(sys.executable)";

/** True when `cmd` looks like a filesystem path rather than a bare command. */
function isFilePath(cmd) {
  return cmd.includes("/") || cmd.includes("\\");
}

/** Platform venv layout: `bin/python` on POSIX, `Scripts/python.exe` on Windows. */
function venvPython(venvRoot) {
  return IS_WIN ? join(venvRoot, "Scripts", "python.exe") : join(venvRoot, "bin", "python");
}

/**
 * @returns {Array<{label:string, argv:string[], cmd:string, extra:string[], checkPath:boolean}>}
 */
function candidates(config) {
  const list = [];
  const seen = new Set();
  const add = (label, cmd, { extra = [], checkPath = false } = {}) => {
    const argv = [cmd, ...extra, "-c", PY_CHECK];
    const key = argv.join("|");
    if (seen.has(key)) return;
    seen.add(key);
    list.push({ label, argv, cmd, extra, checkPath });
  };

  if (config.pythonPath) {
    add("config.pythonPath", config.pythonPath, { checkPath: isFilePath(config.pythonPath) });
  }
  if (process.env.DSH_FINETUNE_PYTHON) {
    add("env DSH_FINETUNE_PYTHON", process.env.DSH_FINETUNE_PYTHON, { checkPath: isFilePath(process.env.DSH_FINETUNE_PYTHON) });
  }
  // Well-known local venvs, in preference order. Skipped automatically when absent.
  add("venv ~/CodeBuddy/.venv", venvPython(join(homedir(), "CodeBuddy", ".venv")), { checkPath: true });
  add("venv ~/.dsh-finetune/venv", venvPython(join(homedir(), ".dsh-finetune", "venv")), { checkPath: true });
  // Interpreters on PATH.
  for (const cmd of IS_WIN ? ["python", "python3"] : ["python3", "python"]) {
    add(`${cmd} on PATH`, cmd);
  }
  if (IS_WIN) add("py -3 launcher", "py", { extra: ["-3"] });
  return list;
}

/**
 * @param {{ pythonPath?: string }} config
 * @returns {{ python: string, info: string } | { python: null, error: string }}
 */
export function resolveTrainPython(config) {
  for (const cand of candidates(config)) {
    if (cand.checkPath && !existsSync(cand.argv[0])) continue;
    const res = spawnSync(cand.argv[0], cand.argv.slice(1), { encoding: "utf8", timeout: 30000 });
    if (res.status !== 0) continue;
    let python = cand.argv[0];
    if (cand.extra.length) {
      // `py` is a launcher, not an interpreter path; resolve the exe it manages
      // so callers can spawn() the returned `python` directly later.
      const exe = spawnSync(cand.argv[0], [...cand.extra, "-c", PY_EXE], { encoding: "utf8", timeout: 30000 });
      if (exe.status === 0 && exe.stdout.trim()) python = exe.stdout.trim();
    }
    return { python, info: res.stdout.trim() };
  }
  const tried = candidates(config)
    .map((cand) => cand.label)
    .join(", ");
  return {
    python: null,
    error: `no Python interpreter with torch+transformers+peft found; tried: ${tried}. ` +
      "Install them into a venv, then set config.pythonPath (or export DSH_FINETUNE_PYTHON).",
  };
}
