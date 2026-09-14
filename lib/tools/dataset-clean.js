import { createReadStream, createWriteStream } from "node:fs";
import { readFile, writeFile, unlink } from "node:fs/promises";
import { createInterface } from "node:readline";
import { resolve, join, dirname } from "node:path";
import { tmpdir } from "node:os";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { defineTool } from "@deepseek-ai/dsh-tools";
import { validateJsonlLines } from "../dataset.js";
import { CleanPipeline, CLEAN_RULE_LABELS } from "../clean.js";
import { resolveTrainPython } from "../python.js";
import { quarantineCsv, quarantineHtml } from "../quarantine.js";

const CLEAN_PY = join(dirname(fileURLToPath(import.meta.url)), "..", "trainer", "clean.py");
const PYTHON_RULES = ["ppl", "lang", "inst", "semantic_dupe", "topic_downsample"];

const RESULT_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    inputLines: { type: "integer" },
    keptLines: { type: "integer" },
    removedLines: { type: "integer" },
    format: { type: "string" },
    byRule: {
      type: "object",
      additionalProperties: false,
      properties: Object.fromEntries(Object.keys(CLEAN_RULE_LABELS).map((rule) => [rule, { type: "integer" }])),
    },
    examples: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          line: { type: "integer" },
          rule: { type: "string" },
          metric: { type: "number" },
          source: { type: "string" },
        },
      },
    },
    outputValid: { type: "boolean" },
    dryRun: { type: "boolean" },
    outputPath: { type: "string" },
    quarantine: { type: "boolean" },
    quarantinePath: { type: "string" },
    reportPath: { type: "string" },
    csvPath: { type: "string" },
    // Null when the optional Python layer is disabled (pyEnabled = false), so the
    // schema must accept null as well as the report object/string.
    pythonReportPath: { oneOf: [{ type: "string" }, { type: "null" }] },
    pythonReport: { oneOf: [{ type: "object", additionalProperties: true }, { type: "null" }] },
  },
};

/** Spawn clean.py, capture stdout+stderr (capped), resolve with exit code. */
function runPython(python, configPath) {
  return new Promise((resolveRun) => {
    const child = spawn(python, [CLEAN_PY, "--config", configPath], {
      env: { ...process.env, HF_HUB_OFFLINE: "1" },
      stdio: ["ignore", "pipe", "pipe"],
    });
    const chunks = [];
    let length = 0;
    const CAP = 200_000;
    const onData = (chunk) => {
      const text = chunk.toString();
      chunks.push(text);
      length += text.length;
      while (length > CAP && chunks.length > 1) length -= chunks.shift().length;
    };
    child.stdout.on("data", onData);
    child.stderr.on("data", onData);
    child.on("error", (error) => resolveRun({ code: -1, output: `spawn failed: ${error.message}` }));
    child.on("close", (code) => resolveRun({ code, output: chunks.join("") }));
  });
}

/** Streaming validation of a written file; an empty output counts as valid. */
async function validatePath(path) {
  const lines = createInterface({ input: createReadStream(path, "utf8"), crlfDelay: Infinity });
  const result = await validateJsonlLines(lines);
  return result.valid || result.lines === 0;
}

export function datasetCleanTool(env = {}) {
  const config = env.config ?? {};
  return defineTool({
    name: "finetune_dataset_clean",
    description:
      "Clean a local JSONL fine-tuning dataset before training: drop records that are invalid / exact duplicates / SimHash near-duplicates / " +
      "too short / too long / repetitive or symbol-heavy, remove control characters, HTML tags, URLs, full-width→half-width, optionally mask PII " +
      "(ID card / phone / email), merge consecutive same-role turns, and enforce that chat/sharegpt records end on an assistant turn. " +
      "Writes a new file (default <path>.clean.jsonl) and never modifies the input; pass dry_run=true to preview the effect without writing. " +
      "Per-source thresholds: per_source=[{match, ...}] overrides thresholds for records whose source tag (record.source, or the input path) " +
      "contains match — e.g. code/math subsets can keep a looser repeat_ratio. Quarantine loop: quarantine=true writes boundary samples to " +
      "<path>.quarantine.jsonl + an HTML/CSV review report for a human Reject/Accept pass before re-tuning thresholds. " +
      "Optional Python layer (uses the bundled lib/trainer/clean.py + local venv, e.g. Qwen3-0.6B): ppl=true filters by perplexity (per-source " +
      "ppl_max supported), lang_check=true drops records whose dominant language is not allowed_langs, instruction_follow=true applies a light " +
      "non-answer heuristic, semantic_dedup=true drops embedding-similar records, topic_downsample=true caps records per topic cluster. " +
      "Run finetune_dataset_validate on the output afterwards to confirm it is trainable.",
    parameters: {
      path: { type: "string", required: true, description: "Absolute path to the .jsonl dataset to clean" },
      out: { type: "string", description: 'Output path for the cleaned dataset; defaults to "<path>.clean.jsonl". The input file is never overwritten.' },
      dry_run: { type: "boolean", description: "Only report what would change; do not write the output file (quarantine files still written when quarantine=true) (default false)" },
      dedupe: { type: "boolean", description: "Drop exact duplicate records (after cleaning) — default true" },
      fuzzy_dedupe: { type: "boolean", description: "Drop SimHash near-duplicates of earlier kept records (rewritten copies that exact dedupe misses) — default true" },
      simhash_distance: { type: "integer", description: "SimHash hamming distance threshold for fuzzy dedupe (default 8; calibrated on CJK bigram tokens: rewrites land at 5–6, same-question/different-answer at 16+; 0 disables the rule even if fuzzy_dedupe is on)" },
      min_chars: { type: "integer", description: "Minimum total characters across all text fields; records below are dropped (default 10; 0 disables)" },
      max_chars: { type: "integer", description: "Maximum total characters across all text fields; records above are dropped (default 20000; 0 disables)" },
      repeat_ratio: { type: "number", description: "Drop records where a single character is more than this fraction of the non-space text (default 0.5; 0 disables)" },
      min_meaningful_ratio: {
        type: "number",
        description: "Drop records where letters/digits are a smaller fraction of the non-space text — catches emoji/symbol garbage (default 0.2; 0 disables)",
      },
      require_assistant_end: { type: "boolean", description: "For chat/sharegpt, drop records that do not end on an assistant turn (default true)" },
      clean_text: { type: "boolean", description: "Clean every text field: full-width→half-width, strip control/zero-width characters, HTML tags and URLs, mask PII, normalise whitespace (default true)" },
      full_width_to_half: { type: "boolean", description: "Convert full-width ASCII (U+FF01–U+FF5E) and full-width space to half-width via mapping table, not NFKC — default true" },
      mask_pii: { type: "boolean", description: "Mask mainland ID-card numbers / mobile phones / emails with [ID]/[PHONE]/[EMAIL] placeholders (default true)" },
      merge_same_role: { type: "boolean", description: "Merge consecutive same-role turns in chat/sharegpt records (default true)" },
      collapse_space: {
        type: "boolean",
        description: "Also collapse runs of spaces/tabs into a single space (default false; off by default to preserve code formatting)",
      },
      per_source: {
        type: "array",
        description: "Per-source threshold overrides: first entry whose match is a substring of the record's source tag (record.source, else the input path) wins",
        items: {
          type: "object",
          additionalProperties: false,
          properties: {
            match: { type: "string", description: "Substring matched (case-insensitive) against the record's source tag" },
            min_chars: { type: "integer", description: "Override min_chars for matching records" },
            max_chars: { type: "integer", description: "Override max_chars for matching records" },
            repeat_ratio: { type: "number", description: "Override repeat_ratio for matching records" },
            min_meaningful_ratio: { type: "number", description: "Override min_meaningful_ratio for matching records" },
            fuzzy_dedupe: { type: "boolean", description: "Override fuzzy_dedupe for matching records" },
            simhash_distance: { type: "integer", description: "Override simhash_distance for matching records" },
            ppl_max: { type: "number", description: "Override the PPL threshold for matching records (only used when ppl=true)" },
          },
        },
      },
      quarantine: { type: "boolean", description: "Write boundary samples (dropped within quarantine_margin of a numeric threshold) to <path>.quarantine.jsonl plus an HTML/CSV review report (default false)" },
      quarantine_margin: { type: "number", description: "Relative deviation band that defines a 'boundary' sample (default 0.25 = 25% above/below threshold)" },
      quarantine_max: { type: "integer", description: "Cap on quarantined records kept for review (default 5000)" },
      // --- optional Python layer (PPL / language / semantics) ---
      ppl: { type: "boolean", description: "Run the Python layer PPL filter: drop records whose perplexity exceeds ppl_max (needs the local torch/transformers venv + a small model) (default false)" },
      ppl_model: { type: "string", description: "HF model id/path for PPL + embeddings (default: plugin defaultLocalModel, e.g. Qwen/Qwen3-0.6B)" },
      ppl_max: { type: "number", description: "Global PPL threshold; records above are dropped (default 150; calibrated on Qwen3-0.6B for >=32-token Chinese QA: p50≈43/p90≈66/p95≈80/p99≈147, so 150 only catches the far tail; per_source.ppl_max overrides per source; tighten after reading the python report quantiles and quarantine samples)" },
      ppl_batch: { type: "integer", description: "Inference batch size for the Python layer (default 32)" },
      ppl_min_tokens: { type: "integer", description: "Records shorter than this many tokens are not PPL-scored (kept): short-text PPL is high-variance and would drop clean one-line QA (default 32)" },
      lang_check: { type: "boolean", description: "Drop records whose dominant language is not in allowed_langs (script-ratio detector, no extra deps) (default false)" },
      allowed_langs: { type: "array", items: { type: "string" }, description: "Languages to keep for lang_check (default [\"zh\", \"en\"])" },
      lang_model: { type: "string", description: "Path to a fasttext lid model (e.g. lid.176.bin). When set, language check uses fasttext predictions instead of the script-share heuristic — needed when the corpus mixes Latin-script languages (en/fr/de/es) that script ratios cannot tell apart; falls back automatically if unavailable" },
      instruction_follow: { type: "boolean", description: "Light non-answer heuristic on the final assistant turn: too-short / question-ending answers are dropped (default false)" },
      semantic_dedup: { type: "boolean", description: "Embedding-based semantic dedup via the model's hidden states + numpy cosine similarity (default false)" },
      semantic_sim: { type: "number", description: "Cosine similarity at or above which two records count as semantic duplicates (default 0.95)" },
      topic_downsample: { type: "boolean", description: "Cap records per topic cluster (numpy k-means on model embeddings) (default false)" },
      topic_max_per_cluster: { type: "integer", description: "Max records kept per topic cluster (default 500)" },
    },
    output: {
      schema: RESULT_SCHEMA,
      render: (_args, value) => [
        {
          type: "text",
          text:
            `dataset cleaned: ${value.inputLines} -> ${value.keptLines} lines (${value.removedLines} removed, format ${value.format}, outputValid ${value.outputValid})` +
            (value.removedLines > 0
              ? ` | by rule: ${Object.entries(value.byRule)
                  .filter(([, count]) => count > 0)
                  .map(([rule, count]) => `${rule}=${count}`)
                  .join(", ")}`
              : "") +
            (value.pythonReport ? ` | python layer: ${PYTHON_RULES.filter((r) => (value.pythonReport.dropped?.[r] ?? 0) > 0).map((r) => `${r}=${value.pythonReport.dropped[r]}`).join(", ") || "ok"}` : "") +
            (value.quarantine ? ` | quarantine: ${value.quarantinePath} + ${value.reportPath}` : "") +
            (value.dryRun ? " | dry run — cleaned output not written" : ` | wrote ${value.outputPath}`),
        },
      ],
    },
    isConcurrencySafe: () => true,
    // Python-layer runs need time for model inference on CPU.
    timeoutMs: 1800000,
    async execute(args, exec) {
      const inPath = args.path;
      const outPath = args.out ?? `${args.path}.clean.jsonl`;
      if (resolve(outPath) === resolve(inPath)) {
        throw new Error("finetune: refusing to overwrite the input file; pass a different --out path");
      }

      const quarantineEnabled = args.quarantine === true;
      const quarantineMax = args.quarantine_max ?? 5000;
      const pyEnabled = [args.ppl, args.lang_check, args.instruction_follow, args.semantic_dedup, args.topic_downsample].some(
        (value) => value === true,
      );
      const quarantinePath = quarantineEnabled ? `${inPath}.quarantine.jsonl` : "";
      const reportPath = quarantineEnabled ? `${inPath}.quarantine.html` : "";
      const csvPath = quarantineEnabled ? `${inPath}.quarantine.csv` : "";
      const pythonReportPath = pyEnabled ? `${inPath}.clean.python-report.json` : "";

      const pipeline = new CleanPipeline({
        dedupe: args.dedupe,
        fuzzyDedupe: args.fuzzy_dedupe,
        simhashDistance: args.simhash_distance,
        minChars: args.min_chars,
        maxChars: args.max_chars,
        repeatRatio: args.repeat_ratio,
        minMeaningfulRatio: args.min_meaningful_ratio,
        requireAssistantEnd: args.require_assistant_end,
        cleanText: args.clean_text,
        mergeSameRole: args.merge_same_role,
        collapseSpace: args.collapse_space,
        fullWidthToHalf: args.full_width_to_half,
        maskPii: args.mask_pii,
        perSource: (args.per_source ?? []).map((entry) => ({
          match: entry.match,
          ...(entry.min_chars !== undefined && { minChars: entry.min_chars }),
          ...(entry.max_chars !== undefined && { maxChars: entry.max_chars }),
          ...(entry.repeat_ratio !== undefined && { repeatRatio: entry.repeat_ratio }),
          ...(entry.min_meaningful_ratio !== undefined && { minMeaningfulRatio: entry.min_meaningful_ratio }),
          ...(entry.fuzzy_dedupe !== undefined && { fuzzyDedupe: entry.fuzzy_dedupe }),
          ...(entry.simhash_distance !== undefined && { simhashDistance: entry.simhash_distance }),
        })),
        quarantine: quarantineEnabled,
        quarantineMargin: args.quarantine_margin,
        quarantineMax,
      });

      // ---- stage-1 pass: stream the input, write kept records as we go (O(1) memory) ----
      const reader = createInterface({ input: createReadStream(inPath, "utf8"), crlfDelay: Infinity });
      const stageWritePath = pyEnabled
        ? join(tmpdir(), `dsh-clean-stage1-${process.pid}-${Date.now()}.jsonl`)
        : args.dry_run
          ? null
          : outPath;
      const writer = stageWritePath ? createWriteStream(stageWritePath) : null;
      let first = true;
      let lineNo = 0;
      const keptOrigLines = []; // original line numbers kept by stage-1, in order
      for await (const raw of reader) {
        if (raw.trim() === "") continue;
        lineNo += 1;
        const result = pipeline.push(raw, lineNo, inPath);
        if (result.action === "keep") {
          keptOrigLines.push(lineNo);
          if (writer) {
            if (!first) writer.write("\n");
            first = false;
            writer.write(JSON.stringify(result.record));
          }
        }
      }
      if (writer) {
        await new Promise((resolveEnd, rejectEnd) => writer.end((error) => (error ? rejectEnd(error) : resolveEnd())));
      }
      const summary = pipeline.summary;

      if (quarantineEnabled) {
        const records = summary.quarantine.records;
        await writeFile(
          quarantinePath,
          records.length > 0 ? `${records.map((entry) => JSON.stringify(entry)).join("\n")}\n` : "",
          "utf8",
        );
      }

      // ---- optional Python layer ----
      let pythonReport = null;
      let pyFinalPath = null;
      if (pyEnabled) {
        const runtime = resolveTrainPython(config);
        if (!runtime.python) {
          await unlink(stageWritePath).catch(() => {});
          throw new Error(`finetune: Python cleaning requested but ${runtime.error}`);
        }
        pyFinalPath = args.dry_run ? join(tmpdir(), `dsh-clean-final-${process.pid}-${Date.now()}.jsonl`) : outPath;
        const cfgPath = join(tmpdir(), `dsh-clean-py-${process.pid}-${Date.now()}.json`);
        const pyConfig = {
          model_path: args.ppl_model || config.defaultLocalModel || "Qwen/Qwen3-0.6B",
          data_path: stageWritePath,
          out_path: pyFinalPath,
          report_path: pythonReportPath,
          source_hint: inPath, // per-source matching must see the ORIGINAL input path, not the temp stage file
          device: "auto",
          dtype: "auto",
          batch_size: args.ppl_batch ?? 32,
          ppl: args.ppl === true,
          ppl_max: args.ppl_max,
          ppl_min_tokens: args.ppl_min_tokens,
          per_source: (args.per_source ?? [])
            .filter((entry) => entry.ppl_max !== undefined)
            .map((entry) => ({ match: entry.match, ppl_max: entry.ppl_max })),
          lang: args.lang_check === true,
          allowed_langs: args.allowed_langs ?? ["zh", "en"],
          lang_model: args.lang_model ?? "",
          instruction_follow: args.instruction_follow === true,
          semantic_dedup: args.semantic_dedup === true,
          semantic_sim: args.semantic_sim,
          topic_downsample: args.topic_downsample === true,
          topic_max_per_cluster: args.topic_max_per_cluster,
          quarantine_path: quarantineEnabled ? quarantinePath : "",
          quarantine_margin: args.quarantine_margin,
          quarantine_max: quarantineMax,
        };
        await writeFile(cfgPath, JSON.stringify(pyConfig, null, 2), "utf8");
        try {
          const { code, output } = await runPython(runtime.python, cfgPath);
          if (code !== 0) {
            throw new Error(`finetune: Python clean failed (exit ${code}): ${output.trim().slice(-1500)}`);
          }
          pythonReport = JSON.parse(await readFile(pythonReportPath, "utf8"));
          for (const rule of PYTHON_RULES) summary.byRule[rule] = pythonReport.dropped?.[rule] ?? 0;
          summary.inputLines = pythonReport.total;
          summary.keptLines = pythonReport.kept;
          summary.removedLines = pythonReport.total - pythonReport.kept;
          // Python reports positions in the stage-1 file; remap to ORIGINAL line numbers
          // so examples speak the same numbering as the JS stage-1 examples.
          for (const example of (pythonReport.examples ?? []).slice(0, 10)) {
            const pos = example.line;
            if (Number.isInteger(pos) && pos >= 1 && pos <= keptOrigLines.length) {
              example.line = keptOrigLines[pos - 1];
            }
            summary.examples.push(example);
          }
        } finally {
          await unlink(cfgPath).catch(() => {});
          await unlink(stageWritePath).catch(() => {});
        }
      }

      // ---- output validation (the "clean output stays trainable" guarantee) ----
      let outputValid;
      if (args.dry_run) {
        outputValid = pyEnabled && pyFinalPath ? await validatePath(pyFinalPath) : summary.outputValid;
      } else {
        outputValid = await validatePath(outPath);
      }
      if (args.dry_run && pyFinalPath) await unlink(pyFinalPath).catch(() => {});

      // ---- quarantine report (after the Python pass so PPL boundary rows are included) ----
      if (quarantineEnabled) {
        let records = summary.quarantine.records;
        if (pyEnabled) {
          records = [];
          const ql = createInterface({ input: createReadStream(quarantinePath, "utf8"), crlfDelay: Infinity });
          for await (const line of ql) {
            if (line.trim() === "") continue;
            try {
              const entry = JSON.parse(line);
              // Python-appended rows carry stage-1 line numbers; remap to original.
              if (PYTHON_RULES.includes(entry.rule) && Number.isInteger(entry.line) && entry.line >= 1 && entry.line <= keptOrigLines.length) {
                entry.line = keptOrigLines[entry.line - 1];
              }
              records.push(entry);
            } catch {
              // ignore malformed rows
            }
            if (records.length >= quarantineMax) break;
          }
          summary.quarantine = { enabled: true, count: records.length, records };
        }
        await writeFile(reportPath, quarantineHtml(records, summary, { path: inPath }), "utf8");
        await writeFile(csvPath, quarantineCsv(records), "utf8");
      }

      return {
        inputLines: summary.inputLines,
        keptLines: summary.keptLines,
        removedLines: summary.removedLines,
        format: summary.format,
        byRule: summary.byRule,
        examples: summary.examples.slice(0, 20),
        outputValid,
        dryRun: args.dry_run === true,
        outputPath: args.dry_run ? "" : outPath,
        quarantine: quarantineEnabled,
        quarantinePath,
        reportPath,
        csvPath,
        pythonReportPath,
        pythonReport,
      };
    },
  });
}
