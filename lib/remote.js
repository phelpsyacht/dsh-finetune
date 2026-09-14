/**
 * SSH transport + remote training runner for dsh-finetune.
 *
 * Purpose: let `finetune_train` run the bundled Python trainer on a remote
 * GPU box (e.g. a Linux server the model weights live on) while the harness
 * keeps running on the local machine.
 *
 * Model architecture mirrors `lib/python.js` + the local `spawnTrainer` path:
 * a `SshTrainRunner` exposes the same surface local training uses
 * (`done` / `kill` / `readOutput`), so `lib/tools/train.js` registers it into
 * the same background-jobs registry without any tool-level changes.
 *
 * Flow per run:
 *   1. mkdir remote run dir
 *   2. rsync up: dataset + config.json + bundled train.py
 *   3. launch remotely: `nohup python train.py ... & echo $! > pid`
 *   4. poll (one ssh round trip per tick): pid alive? exitcode file?
 *      training_state.json? incremental log tail
 *   5. resolve `done` on terminal state; `kill` = TERM pid + pkill fallback
 *
 * Credentials are configured (SSH key path, host, user), never tool args.
 */
import { spawn } from "node:child_process";
import { basename } from "node:path";

/** POSIX single-quote a value for remote `sh -c` strings. */
export function shq(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function sshOptions(remote) {
  // `accept-new` is the pragmatic default: fails on known-key mismatch but
  // records unknown hosts non-interactively (BatchMode). Set strictHostKey
  // true for a hard verify.
  const strict = remote.strictHostKey === true ? "yes" : "accept-new";
  const args = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=15",
    "-o", `StrictHostKeyChecking=${strict}`,
  ];
  if (remote.keyPath) args.push("-i", remote.keyPath);
  if (remote.port && Number(remote.port) > 0) args.push("-p", String(remote.port));
  return args;
}

/** argv for `ssh <options> <user>@<host> <command>`. */
export function sshCommand(remote, command) {
  return [...sshOptions(remote), `${remote.user}@${remote.host}`, command];
}

/**
 * Spawn a binary, capture stdout/stderr, never throw for nonzero exits.
 * @returns {Promise<{ code: number, stdout: string, stderr: string }>}
 */
export function execFileAsync(bin, args, { timeoutMs = 60000, maxBuffer = 64 * 1024 * 1024 } = {}) {
  return new Promise((resolve) => {
    let stdout = "";
    let stderr = "";
    let settled = false;
    const settle = (value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(value);
    };
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      settle({ code: -1, stdout, stderr: `${stderr}\n[exec timed out after ${timeoutMs}ms]`.trim() });
    }, timeoutMs);
    const child = spawn(bin, args, { stdio: ["ignore", "pipe", "pipe"] });
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
      if (stdout.length > maxBuffer) child.kill("SIGKILL");
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", (error) => {
      settle({ code: -1, stdout, stderr: `${stderr}\n[spawn error: ${error.message}]`.trim() });
    });
    child.on("close", (code) => {
      settle({ code: code ?? -1, stdout, stderr });
    });
  });
}

/** Run one command on the remote host. */
export function remoteExec(remote, command, options) {
  return execFileAsync("ssh", sshCommand(remote, command), options);
}

function rsyncInvocation(remote) {
  const ssh = ["ssh", ...sshOptions(remote)].map(shq).join(" ");
  return ["-az", "--partial", "-e", ssh];
}

/** rsync one or more local files into an existing remote directory. */
export function remoteUpload(remote, sources, destDir, options) {
  return execFileAsync(
    "rsync",
    [...rsyncInvocation(remote), ...sources, `${remote.user}@${remote.host}:${destDir}/`],
    options,
  ).then((res) => {
    // Windows does not ship rsync; make the failure actionable instead of a
    // bare "spawn rsync ENOENT".
    if (res.code === -1 && process.platform === "win32" && /rsync|ENOENT/i.test(res.stderr)) {
      res.stderr +=
        "\n[hint] rsync is not bundled with Windows. Install it (Git for Windows / MSYS2 / WSL) and add it to PATH, or run the plugin inside WSL.";
    }
    return res;
  });
}

export function remoteMkdir(remote, dir, options) {
  return remoteExec(remote, `mkdir -p ${shq(dir)}`, options);
}

/** Resolve the remote user's home dir so default paths stay shell-safe. */
export async function remoteHomeDir(remote) {
  const res = await remoteExec(remote, 'printf "%s" "$HOME"', { timeoutMs: 30000 });
  if (res.code !== 0) {
    throw new Error(`finetune: cannot resolve remote $HOME: ${res.stderr || res.stdout}`);
  }
  const home = res.stdout.trim();
  if (!home) throw new Error("finetune: remote $HOME resolved to empty string");
  return home;
}

const MAX_CONSECUTIVE_ERRORS = 3;

/**
 * Remote (SSH) counterpart of the local spawnTrainer runner.
 *
 * Starts training immediately on creation. Exposes:
 * - `done: Promise<{status, detail, output}>`   terminal state of the run
 * - `kill(): void`                              TERM the remote process
 * - `readOutput(): string`                      incremental buffered log
 *
 * @param {object} options
 * @param {object} options.remote            ssh config ({host,user,port,keyPath,strictHostKey})
 * @param {string} options.runId
 * @param {string} options.python            remote interpreter (path or command)
 * @param {string} options.localTrainPy      local path of the bundled train.py (uploaded)
 * @param {string} options.localConfigPath   local path of run config.json (uploaded)
 * @param {string} options.datasetPath       local path of the JSONL dataset (uploaded)
 * @param {string} options.runDir            absolute remote run directory
 * @param {number} [options.pollIntervalMs]  seconds-between-probes in ms
 */
export function createSshTrainRunner(options) {
  const {
    remote,
    runId,
    python,
    localTrainPy,
    localConfigPath,
    datasetPath,
    runDir,
    pollIntervalMs = 60000,
  } = options;

  const datasetName = basename(datasetPath);
  const cfgRemote = `${runDir}/config.json`;
  const trainPyRemote = `${runDir}/train.py`;
  const logRemote = `${runDir}/train.log`;
  const pidFile = `${runDir}/pid`;
  const exitFile = `${runDir}/exitcode`;
  const stateFile = `${runDir}/training_state.json`;
  const hostLabel = `${remote.user}@${remote.host}`;

  const lines = [];
  let cursor = 0;
  let byteOffset = 0;
  let stopped = false;
  let currentPid = "";
  let doneResolve;
  const done = new Promise((resolve) => {
    doneResolve = resolve;
  });
  const log = (text) => lines.push(text);

  async function launch() {
    await remoteMkdir(remote, runDir, { timeoutMs: 30000 });
    const upload = await remoteUpload(
      remote,
      [localConfigPath, datasetPath, localTrainPy],
      runDir,
      { timeoutMs: 180000 },
    );
    if (upload.code !== 0) {
      throw new Error(`upload to ${hostLabel}:${runDir} failed: ${(upload.stderr || upload.stdout).slice(0, 400)}`);
    }
    // Launch detached on the remote side. All of the child's stdio is
    // redirected, so ssh returns right after echoing the pid into `pid`.
    const launchCmd = [
      `{ nohup ${shq(python)} ${shq(trainPyRemote)} --config ${shq(cfgRemote)} > ${shq(logRemote)} 2>&1 < /dev/null & echo $! > ${shq(pidFile)}; }`,
      `cat ${shq(pidFile)}`,
    ].join("; ");
    const run = await remoteExec(remote, launchCmd, { timeoutMs: 60000 });
    if (run.code !== 0) {
      throw new Error(`remote launch failed on ${hostLabel}: ${(run.stderr || run.stdout).slice(0, 400)}`);
    }
    const pid = run.stdout.trim().split(/\s+/).pop() ?? "";
    currentPid = pid;
    log(`[${runId}] ssh ${hostLabel}: launched pid=${pid} (log ${logRemote})`);
  }

  /** One ssh round trip: alive? exitcode? state file? incremental log. */
  async function probe() {
    const script = [
      `cd ${shq(runDir)} 2>/dev/null || exit 9`,
      `[ -f ${shq(pidFile)} ] && { kill -0 "$(cat ${shq(pidFile)})" 2>/dev/null && echo "__PIDSTATE=alive"; } || true`,
      `[ -f ${shq(exitFile)} ] && echo "__EXIT=$(cat ${shq(exitFile)})"`,
      `[ -f ${shq(stateFile)} ] && echo "__STATE=$(tr -d '\\n' < ${shq(stateFile)})"`,
      `[ -f ${shq(logRemote)} ] && { echo "__BYTES=$(wc -c < ${shq(logRemote)})"; echo "__LOG"; tail -c +${byteOffset + 1} ${shq(logRemote)}; }`,
    ].join("; ");
    return remoteExec(remote, script, { timeoutMs: 30000 });
  }

  async function loop() {
    try {
      await launch();
    } catch (error) {
      log(`launch error: ${error.message}`);
      doneResolve({ status: "failed", detail: `remote launch failed: ${error.message}`, output: lines.join("\n") });
      return;
    }

    let consecutiveErrors = 0;
    for (;;) {
      if (stopped) {
        doneResolve({ status: "killed", detail: `ssh run ${runId} cancelled`, output: lines.join("\n") });
        return;
      }
      let result;
      try {
        result = await probe();
        consecutiveErrors = 0;
      } catch (error) {
        if (stopped) {
          doneResolve({ status: "killed", detail: `ssh run ${runId} cancelled`, output: lines.join("\n") });
          return;
        }
        consecutiveErrors += 1;
        log(`probe error: ${error.message}`);
        if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
          doneResolve({
            status: "failed",
            detail: `probing ${hostLabel} failed ${MAX_CONSECUTIVE_ERRORS} times in a row`,
            output: lines.join("\n"),
          });
          return;
        }
        await sleep(pollIntervalMs);
        continue;
      }

      if (result.code === 9) {
        doneResolve({ status: "failed", detail: `remote run dir vanished: ${runDir}`, output: lines.join("\n") });
        return;
      }
      if (result.code !== 0) {
        log(`probe shell exit ${result.code}: ${(result.stderr || result.stdout).slice(0, 400)}`);
      }
      const stdout = result.stdout;

      // Consume the incremental log delta (__LOG marker last in the script).
      const logPart = stdout.split("__LOG")[1];
      if (logPart !== undefined) {
        const chunk = logPart.replace(/^\n/, "");
        if (chunk) log(chunk.replace(/\n$/, ""));
      }
      const bytes = /__BYTES=(\d+)/.exec(stdout);
      if (bytes) byteOffset = Math.max(byteOffset, Number(bytes[1]));

      const exit = /__EXIT=(\S+)/.exec(stdout);
      if (exit) {
        const state = /__STATE=(.*)$/m.exec(stdout)?.[1] ?? "";
        if (exit[1] === "0") {
          log(`final: ${state || "no training_state.json"}`);
          doneResolve({
            status: "completed",
            detail: state ? `finished: ${state}` : "finished (exit 0)",
            output: lines.join("\n"),
          });
        } else {
          doneResolve({
            status: "failed",
            detail: `exit code ${exit[1]} — see ${logRemote}`,
            output: lines.join("\n"),
          });
        }
        return;
      }

      if (!/__PIDSTATE=alive/.test(stdout)) {
        log("remote process no longer alive before an exitcode was written");
        doneResolve({
          status: "failed",
          detail: "remote training process exited unexpectedly (no exitcode found)",
          output: lines.join("\n"),
        });
        return;
      }

      await sleep(pollIntervalMs);
    }
  }

  // Start immediately; the caller binds `done`/`readOutput`/`kill` into the
  // background-jobs registry (or polls the remote log when none exists).
  loop();

  return {
    done,
    kill: () => {
      stopped = true;
      const cmd = [
        `[ -f ${shq(pidFile)} ] && kill -TERM "$(cat ${shq(pidFile)})" 2>/dev/null`,
        `pkill -f ${shq(trainPyRemote)} 2>/dev/null`,
        "true",
      ].join("; ");
      remoteExec(remote, cmd, { timeoutMs: 30000 }).catch(() => {});
    },
    readOutput: () => {
      const out = lines.slice(cursor).join("\n");
      cursor = lines.length;
      return out;
    },
    get pid() {
      return currentPid;
    },
  };
}
