/**
 * OpenAI-compatible fine-tuning REST adapter.
 *
 * Endpoint shape follows the de-facto standard surface adopted by most
 * providers (`POST /files` multipart upload, `POST /fine_tuning/jobs`,
 * `GET /fine_tuning/jobs/:id`, `GET /fine_tuning/jobs`, `POST
 * /fine_tuning/jobs/:id/cancel`). If the target provider deviates, change this
 * file only — the tools consume the normalized objects below.
 */
import { readFile } from "node:fs/promises";
import { basename } from "node:path";

/** Statuses after which no further polling is useful. */
const TERMINAL_STATUSES = new Set(["succeeded", "failed", "cancelled", "canceled"]);

export class FinetuneProvider {
  /**
   * @param {{ baseURL: string, apiKeyEnv?: string, apiKey?: string, requestTimeoutMs?: number }} options
   * The key is resolved lazily per request: a missing key must surface as a
   * tool-call error, never as a boot failure of the whole harness.
   */
  constructor({ baseURL, apiKeyEnv = "DEEPSEEK_API_KEY", apiKey, requestTimeoutMs = 120000 }) {
    this.baseURL = baseURL.replace(/\/+$/, "");
    this.apiKeyEnv = apiKeyEnv;
    this.#apiKey = apiKey;
    this.requestTimeoutMs = requestTimeoutMs;
  }

  #apiKey;

  #resolveApiKey() {
    if (this.#apiKey === undefined) this.#apiKey = resolveApiKey({ apiKeyEnv: this.apiKeyEnv });
    return this.#apiKey;
  }

  async #request(path, { method = "GET", body, formData, signal } = {}) {
    const apiKey = this.#resolveApiKey();
    const controller = new AbortController();
    const timer = setTimeout(
      () => controller.abort(new Error(`finetune: ${path} timed out after ${this.requestTimeoutMs}ms`)),
      this.requestTimeoutMs,
    );
    const onCallerAbort = () => controller.abort(signal.reason);
    signal?.addEventListener("abort", onCallerAbort, { once: true });
    try {
      const headers = { Authorization: `Bearer ${apiKey}` };
      if (!formData && body !== undefined) headers["Content-Type"] = "application/json";
      const res = await fetch(new URL(path, this.baseURL), {
        method,
        headers,
        body: formData ?? (body !== undefined ? JSON.stringify(body) : undefined),
        signal: controller.signal,
      });
      const text = await res.text();
      if (!res.ok) {
        throw new Error(`finetune: HTTP ${res.status} on ${path}: ${text.slice(0, 500)}`);
      }
      return text ? JSON.parse(text) : {};
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onCallerAbort);
    }
  }

  /**
   * Upload a JSONL training file (purpose `fine-tune`) and return its id.
   * @param {string} filePath
   * @param {AbortSignal} [signal]
   */
  async uploadDataset(filePath, signal) {
    const content = await readFile(filePath, "utf8");
    const form = new FormData();
    form.append("purpose", "fine-tune");
    form.append("file", new File([content], basename(filePath), { type: "application/jsonl" }));
    const raw = await this.#request("files", { method: "POST", formData: form, signal });
    const id = raw.id ?? raw.file_id;
    if (!id) throw new Error(`finetune: upload returned no file id: ${JSON.stringify(raw).slice(0, 300)}`);
    return { id: String(id), filename: basename(filePath), bytes: Buffer.byteLength(content) };
  }

  /**
   * Create a fine-tuning job for an uploaded training file.
   * @param {{ trainingFile: string, model: string, suffix?: string, hyperparameters?: object }} spec
   * @param {AbortSignal} [signal]
   */
  async createJob(spec, signal) {
    const raw = await this.#request("fine_tuning/jobs", {
      method: "POST",
      body: {
        training_file: spec.trainingFile,
        model: spec.model,
        ...(spec.suffix !== undefined && { suffix: spec.suffix }),
        ...(spec.hyperparameters !== undefined && { hyperparameters: spec.hyperparameters }),
      },
      signal,
    });
    return this.#normalizeJob(raw);
  }

  /** @param {string} jobId @param {AbortSignal} [signal] */
  async getJob(jobId, signal) {
    return this.#normalizeJob(await this.#request(`fine_tuning/jobs/${encodeURIComponent(jobId)}`, { signal }));
  }

  /** @param {number} [limit] @param {AbortSignal} [signal] */
  async listJobs(limit = 10, signal) {
    const raw = await this.#request(`fine_tuning/jobs?limit=${limit}`, { signal });
    const rows = Array.isArray(raw) ? raw : (raw.data ?? []);
    return rows.map((row) => this.#normalizeJob(row));
  }

  /** @param {string} jobId @param {AbortSignal} [signal] */
  async cancelJob(jobId, signal) {
    return this.#normalizeJob(
      await this.#request(`fine_tuning/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST", signal }),
    );
  }

  #normalizeJob(raw) {
    const id = raw.id ?? raw.fine_tuning_job_id;
    if (id === undefined) throw new Error(`finetune: provider returned a job without an id: ${JSON.stringify(raw).slice(0, 300)}`);
    return {
      jobId: String(id),
      status: String(raw.status ?? "unknown").toLowerCase(),
      terminal: TERMINAL_STATUSES.has(String(raw.status ?? "").toLowerCase()),
      model: raw.model ?? "",
      fineTunedModel: raw.fine_tuned_model ?? raw.fineTunedModel ?? "",
      createdAt: raw.created_at !== undefined ? String(raw.created_at) : "",
      detail: raw.error?.message ?? raw.failed_error ?? "",
    };
  }
}

/**
 * Resolve the API key from the environment. Deliberately never a tool
 * parameter: keys in tool arguments would land in the model-visible session
 * transcript.
 * @param {{ apiKeyEnv?: string }} config
 */
export function resolveApiKey(config) {
  const envName = config.apiKeyEnv ?? "DEEPSEEK_API_KEY";
  const key = process.env[envName];
  if (!key) {
    throw new Error(
      `finetune: no API key; export ${envName} or set apiKeyEnv in the plugin config row (never pass keys through tool arguments)`,
    );
  }
  return key;
}

export { TERMINAL_STATUSES };
