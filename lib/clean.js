/**
 * Data cleaning for JSONL fine-tuning datasets.
 *
 * Runs on top of `lib/dataset.js` format detection (chat / prompt-completion /
 * alpaca / sharegpt). Cleaning is split into two layers:
 *
 *  1. Per-record transformations (in-place, only when enabled):
 *     - `cleanText`   full-width→half-width (mapping table, NOT NFKC), strip
 *                     control / zero-width / bidi characters, HTML tags and
 *                     URLs; optionally mask PII (ID card / mainland phone /
 *                     email); normalise newlines; optionally collapse space
 *                     runs (off by default to preserve code formatting).
 *     - `mergeSameRole` merge consecutive same-role turns in chat / sharegpt.
 *  2. Per-record filters (a record is dropped on its FIRST failing rule, so the
 *     per-rule counts in `byRule` are mutually exclusive):
 *     - invalid             does not parse as JSON / is not a supported format /
 *                           does not match the file's detected format
 *     - empty_after_clean   a required text field became empty after cleanup
 *     - dedupe              canonical JSON duplicate of an earlier kept record
 *     - fuzzy_dupe          SimHash near-duplicate of an earlier kept record
 *                           (runs right after exact dedupe, as the second
 *                           dedupe pass; token-level, 64-bit SimHash with a
 *                           hamming-distance threshold — catches "rewritten"
 *                           near-duplicates that exact dedupe misses)
 *     - min_chars / max_chars  total characters across all text fields
 *     - repeat_ratio        a single character dominates the non-space text
 *     - meaningful_ratio    too few letters/digits (emoji / symbol garbage)
 *     - ends_assistant      chat / sharegpt must end on an assistant turn
 *     - ppl / lang / inst / semantic_dupe / topic_downsample  filled in by the
 *                           optional Python layer (lib/trainer/clean.py) and
 *                           merged back into `byRule` by the tool
 *
 * Threshold rules honour per-source overrides (`perSource`): the first entry
 * whose `match` is a (case-insensitive) substring of the record's source tag
 * (record.source, falling back to the input path passed as `sourceHint`)
 * overrides the shared thresholds for that record — e.g. code/math subsets can
 * keep a looser repeatRatio / maxChars.
 *
 * When `quarantine` is enabled, records dropped by a numeric threshold rule
 * whose metric sits within `quarantineMargin` of the threshold (the "boundary
 * samples") are collected instead of vanishing; the tool writes them to
 * <path>.quarantine.jsonl plus a CSV/HTML review report for a human
 * Reject/Accept pass before re-tuning thresholds.
 *
 * The pipeline (`CleanPipeline`) is line-oriented so the tool can stream a
 * file with O(1) memory; `cleanJsonlDataset` is a thin in-memory wrapper over
 * the same engine (kept for callers that already hold the whole text, e.g. the
 * inline clean pass in `finetune_train`).
 *
 * Design notes (kept on purpose):
 * - Whitelist-style text cleanup: only known-harmful characters/tags/URLs are
 *   removed; chat template markers ([INST], <|im_start|>) are never touched.
 * - collapseSpace stays off by default: it would destroy code indentation.
 * - The original file is never modified: the caller writes the returned `text`
 *   (or the streamed records) to a new path.
 */

import { detectFormat } from "./dataset.js";

// Control chars excluding \n and \t; plus zero-width (U+200B–U+200D, U+FEFF),
// bidi control (U+202A–U+202E), line/paragraph separators and word joiners.
const CONTROL_CHARS_RE =
  /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F\u200B-\u200F\u2028\u2029\u202A-\u202E\u2060-\u206F\uFEFF]/g;
const HTML_TAG_RE = /<[^>]*>/g;
const URL_RE = /https?:\/\/[^\s<>"')\]}]+/gi;
const NEWLINE_RUN_RE = /\n{3,}/g;
const SPACE_RUN_RE = /[ \t]{2,}/g;
// Unicode letters and numbers: covers CJK, Latin, Cyrillic, digits, etc.
const MEANINGFUL_CHAR_RE = /[\p{L}\p{N}]/u;

// ---------------------------------------------------------------------------
// Full-width → half-width
// ---------------------------------------------------------------------------
// The full-width forms of printable ASCII (U+0021–U+007E) live contiguously at
// U+FF01–U+FF5E; U+3000 is the full-width space. We deliberately use a mapping
// table instead of String.normalize("NFKC"): NFKC also decomposes ①→1, ﬁ→fi,
// ㍿→株式会社 etc., which silently rewrites code/math/data content beyond what
// was asked. The table touches only what it must.
const FULL_WIDTH_BASE = 0xff01 - 0x21; // 0xFEE0
const FULL_WIDTH_RE = /[\uFF01-\uFF5E\u3000]/g;

function fullWidthToHalf(value) {
  return value.replace(FULL_WIDTH_RE, (ch) =>
    ch === "\u3000" ? " " : String.fromCharCode(ch.charCodeAt(0) - FULL_WIDTH_BASE),
  );
}

// ---------------------------------------------------------------------------
// PII masking
// ---------------------------------------------------------------------------
const PII_PATTERNS = [
  // 18-digit mainland ID card number (last digit may be X), word-bounded.
  [/\b\d{17}[\dXx]\b/g, "[ID]"],
  // Mainland mobile: 11 digits starting 13–19.
  [/\b1[3-9]\d{9}\b/g, "[PHONE]"],
  // Email addresses.
  [/[\w.+-]+@[\w-]+\.[\w.]+/g, "[EMAIL]"],
];

function maskPii(value) {
  for (const [re, placeholder] of PII_PATTERNS) value = value.replace(re, placeholder);
  return value;
}

// ---------------------------------------------------------------------------
// 64-bit SimHash (near-duplicate / fuzzy dedupe)
// ---------------------------------------------------------------------------
const SIMHASH_BITS = 64;
const SIMHASH_BANDS = 4; // 64 bits → 4 × 16-bit bands
// A band bucket larger than this is treated as "hot" (too many colliding
// candidates to be informative) and skipped, bounding worst-case cost.
const SIMHASH_HOT_BAND = 1000;

/** FNV-1a 64-bit string hash (BigInt). */
function fnv1a64(str) {
  const mask = (1n << 64n) - 1n;
  let hash = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  for (let i = 0; i < str.length; i += 1) {
    hash ^= BigInt(str.charCodeAt(i));
    hash = (hash * prime) & mask;
  }
  return hash;
}

/**
 * Tokenise text for SimHash: CJK character bigrams + lowercased Latin words.
 * Bigrams capture Chinese phrasal structure without a segmenter.
 */
export function simhashTokens(text) {
  const tokens = [];
  const cjk = text.match(/[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/g);
  if (cjk) {
    for (let i = 0; i < cjk.length; i += 1) {
      tokens.push(i + 1 < cjk.length ? `${cjk[i]}${cjk[i + 1]}` : cjk[i]);
    }
  }
  const latin = text.toLowerCase().match(/[a-z0-9]+/g);
  if (latin) tokens.push(...latin);
  return tokens;
}

/** 64-bit SimHash of a token list (token frequency = weight). */
export function simhash64(tokens) {
  const weights = new Map();
  for (const token of tokens) weights.set(token, (weights.get(token) ?? 0) + 1);
  const votes = new Array(SIMHASH_BITS).fill(0);
  for (const [token, weight] of weights) {
    const hash = fnv1a64(token);
    for (let bit = 0; bit < SIMHASH_BITS; bit += 1) {
      votes[bit] += ((hash >> BigInt(bit)) & 1n) === 1n ? weight : -weight;
    }
  }
  let hash = 0n;
  for (let bit = 0; bit < SIMHASH_BITS; bit += 1) {
    if (votes[bit] > 0) hash |= 1n << BigInt(bit);
  }
  return hash;
}

/** Hamming distance between two 64-bit SimHashes. */
export function hammingDistance(a, b) {
  let x = a ^ b;
  let count = 0;
  while (x) {
    count += Number(x & 1n);
    x >>= 1n;
  }
  return count;
}

/**
 * Incremental near-duplicate index. Bands of 16 bits index candidate buckets;
 * only hashes sharing a band are compared with the full hamming distance, so
 * lookups stay sub-linear for typical dataset sizes.
 */
export class SimhashIndex {
  constructor(distance = 3) {
    this.distance = distance;
    this.hashes = [];
    this.bands = Array.from({ length: SIMHASH_BANDS }, () => new Map());
  }

  add(hash) {
    const index = this.hashes.length;
    this.hashes.push(hash);
    for (let band = 0; band < SIMHASH_BANDS; band += 1) {
      const key = Number((hash >> BigInt(band * 16)) & 0xffffn);
      const bucket = this.bands[band].get(key);
      if (bucket) bucket.push(index);
      else this.bands[band].set(key, [index]);
    }
  }

  isNearDuplicate(hash) {
    if (this.hashes.length === 0) return false;
    const seen = new Set();
    for (let band = 0; band < SIMHASH_BANDS; band += 1) {
      const key = Number((hash >> BigInt(band * 16)) & 0xffffn);
      const bucket = this.bands[band].get(key);
      if (!bucket || bucket.length > SIMHASH_HOT_BAND) continue;
      for (const index of bucket) {
        if (seen.has(index)) continue;
        seen.add(index);
        if (hammingDistance(hash, this.hashes[index]) <= this.distance) return true;
      }
    }
    return false;
  }
}

// ---------------------------------------------------------------------------
// Rule labels & defaults
// ---------------------------------------------------------------------------

export const CLEAN_DEFAULTS = {
  dedupe: true,
  fuzzyDedupe: true,
  simhashDistance: 8, // hamming distance ≤ this = near-duplicate (0 disables)
  // Calibrated on CJK bigram tokens: a 2–8 token rewrite lands at distance
  // 5–6, same-question/different-answer at 16+, unrelated topics at 23+.
  simhashMinTokens: 10, // short records are never fuzzy-checked (false positives)
  minChars: 10, // 0 disables
  maxChars: 20000, // 0 disables
  repeatRatio: 0.5, // 0 disables
  minMeaningfulRatio: 0.2, // 0 disables
  requireAssistantEnd: true,
  cleanText: true,
  mergeSameRole: true,
  collapseSpace: false,
  fullWidthToHalf: true,
  maskPii: true,
  perSource: [], // [{ match, minChars?, maxChars?, repeatRatio?, minMeaningfulRatio?, fuzzyDedupe?, simhashDistance? }]
  quarantine: false,
  quarantineMargin: 0.25, // relative deviation band for "boundary" samples
  quarantineMax: 5000,
};

export const CLEAN_RULE_LABELS = {
  invalid: "invalid",
  empty_after_clean: "empty_after_clean",
  dedupe: "dedupe",
  fuzzy_dupe: "fuzzy_dupe",
  min_chars: "min_chars",
  max_chars: "max_chars",
  repeat_ratio: "repeat_ratio",
  meaningful_ratio: "meaningful_ratio",
  ends_assistant: "ends_assistant",
  // Python layer rules (lib/trainer/clean.py); the tool merges their counts in.
  ppl: "ppl",
  lang: "lang",
  inst: "inst",
  semantic_dupe: "semantic_dupe",
  topic_downsample: "topic_downsample",
};

/** Direction each numeric rule fails on: "over" (metric above threshold) or "under". */
const RULE_DIRECTION = {
  min_chars: "under",
  max_chars: "over",
  repeat_ratio: "over",
  meaningful_ratio: "under",
  ppl: "over",
};

/** [owner, key] pairs of every text field of a record, in field order. */
function textFieldRefs(record, format) {
  if (format === "chat") return record.messages.map((message) => [message, "content"]);
  if (format === "sharegpt") return record.conversations.map((turn) => [turn, "value"]);
  if (format === "prompt-completion") {
    return [
      [record, "prompt"],
      [record, "completion"],
    ];
  }
  if (format === "alpaca") {
    const refs = [[record, "instruction"]];
    if (typeof record.input === "string") refs.push([record, "input"]);
    refs.push([record, "output"]);
    return refs;
  }
  return [];
}

/** [owner, key] pairs of the fields that must stay non-empty. */
function requiredFieldRefs(record, format) {
  if (format === "chat") return record.messages.map((message) => [message, "content"]);
  if (format === "sharegpt") return record.conversations.map((turn) => [turn, "value"]);
  if (format === "prompt-completion") {
    return [
      [record, "prompt"],
      [record, "completion"],
    ];
  }
  if (format === "alpaca") {
    return [
      [record, "instruction"],
      [record, "output"],
    ];
  }
  return [];
}

function cleanTextValue(value, opts) {
  let text = value;
  if (opts.fullWidthToHalf) text = fullWidthToHalf(text);
  text = text
    .replace(/\r\n?/g, "\n") // normalise CRLF / lone CR to LF
    .replace(CONTROL_CHARS_RE, "")
    .replace(HTML_TAG_RE, "")
    .replace(URL_RE, "");
  if (opts.maskPii) text = maskPii(text);
  if (opts.collapseSpace) text = text.replace(SPACE_RUN_RE, " ");
  return text.replace(NEWLINE_RUN_RE, "\n\n").trim();
}

/** Merge consecutive turns that share the same chat role (chat / sharegpt). */
function mergeSameRole(record, format) {
  const roleOfSharegpt = (from) =>
    from === "human" || from === "user" ? "user" : from === "gpt" || from === "assistant" ? "assistant" : "system";
  const field = format === "chat" ? "content" : "value";
  const items = format === "chat" ? record.messages : record.conversations;
  const roleOf = format === "chat" ? (item) => item.role : (item) => roleOfSharegpt(item.from);
  const merged = [];
  for (const item of items) {
    const last = merged[merged.length - 1];
    if (last && roleOf(last) === roleOf(item)) last[field] += `\n${item[field]}`;
    else merged.push(item);
  }
  if (format === "chat") record.messages = merged;
  else record.conversations = merged;
}

/** Stable key for dedupe: JSON with object keys sorted recursively. */
function canonicalKey(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalKey).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const keys = Object.keys(value).sort();
    return `{${keys.map((key) => `${JSON.stringify(key)}:${canonicalKey(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

/** Highest single-character share of the non-space text across all fields. */
function worstRepeatRatio(texts) {
  let worst = 0;
  for (const text of texts) {
    const compact = text.replace(/\s+/g, "");
    // Short fields (e.g. a one-word answer like "y" or "好") always have a
    // dominant character; the rule targets real repetition spam, so only
    // evaluate fields long enough for repetition to be meaningful.
    if (compact.length < 5) continue;
    const counts = new Map();
    for (const ch of compact) counts.set(ch, (counts.get(ch) ?? 0) + 1);
    const ratio = Math.max(...counts.values()) / compact.length;
    if (ratio > worst) worst = ratio;
  }
  return worst;
}

/** Share of letters/digits in the non-space text across all fields. */
function meaningfulRatio(texts) {
  let meaningful = 0;
  let total = 0;
  for (const text of texts) {
    const compact = text.replace(/\s+/g, "");
    total += compact.length;
    for (const ch of compact) if (MEANINGFUL_CHAR_RE.test(ch)) meaningful += 1;
  }
  return total === 0 ? 0 : meaningful / total;
}

function endsWithAssistant(record, format) {
  if (format === "chat") return record.messages[record.messages.length - 1].role === "assistant";
  if (format === "sharegpt") {
    const lastFrom = record.conversations[record.conversations.length - 1].from;
    return lastFrom === "gpt" || lastFrom === "assistant";
  }
  return true; // prompt-completion / alpaca always end on the answer field
}

/**
 * Per-record effective options: base options overridden by the first
 * `perSource` entry whose `match` is a case-insensitive substring of `source`.
 */
function resolveEffectiveOpts(base, source) {
  const perSource = base.perSource;
  if (!source || !Array.isArray(perSource) || perSource.length === 0) return base;
  const needle = String(source).toLowerCase();
  for (const entry of perSource) {
    if (!entry || typeof entry.match !== "string" || entry.match.length === 0) continue;
    if (!needle.includes(entry.match.toLowerCase())) continue;
    const merged = { ...base };
    for (const key of Object.keys(entry)) {
      if (key === "match" || entry[key] === undefined) continue;
      merged[key] = entry[key];
    }
    return merged;
  }
  return base;
}

/**
 * Line-oriented cleaning engine. Feed raw JSONL lines (non-empty) one at a
 * time; kept records come back in `push` results and can be serialised to the
 * output stream immediately, so memory stays O(1) regardless of file size.
 */
export class CleanPipeline {
  constructor(options = {}) {
    this.baseOpts = { ...CLEAN_DEFAULTS };
    for (const [key, value] of Object.entries(options)) {
      if (value !== undefined) this.baseOpts[key] = value;
    }
    this.reset();
  }

  reset() {
    this.format = null;
    this.byRule = Object.fromEntries(Object.keys(CLEAN_RULE_LABELS).map((rule) => [rule, 0]));
    this.examples = [];
    this.keptCount = 0;
    this.inputCount = 0;
    this.seen = new Set();
    this.simhashIndex = new SimhashIndex(this.baseOpts.simhashDistance);
    this.quarantineRecords = [];
    this.bySource = new Map();
  }

  #trackSource(source, kept) {
    const key = source || "(none)";
    const entry = this.bySource.get(key) ?? { count: 0, kept: 0, removed: 0 };
    entry.count += 1;
    if (kept) entry.kept += 1;
    else entry.removed += 1;
    this.bySource.set(key, entry);
  }

  #drop(rule, lineNo, source) {
    this.byRule[rule] += 1;
    if (this.examples.length < 10) this.examples.push({ line: lineNo, rule });
    this.#trackSource(source, false);
  }

  /** Drop by a numeric threshold rule; collect boundary samples when quarantine is on. */
  #dropNumeric(rule, lineNo, source, metric, threshold, record, opts) {
    this.byRule[rule] += 1;
    if (this.examples.length < 10) this.examples.push({ line: lineNo, rule });
    this.#trackSource(source, false);
    const direction = RULE_DIRECTION[rule];
    if (
      opts.quarantine &&
      direction &&
      Number.isFinite(metric) &&
      Number.isFinite(threshold) &&
      threshold > 0 &&
      this.quarantineRecords.length < opts.quarantineMax
    ) {
      const deviation =
        direction === "over" ? (metric - threshold) / threshold : (threshold - metric) / threshold;
      if (deviation > 0 && deviation <= opts.quarantineMargin) {
        this.quarantineRecords.push({
          line: lineNo,
          rule,
          source: source || null,
          metric: Math.round(metric * 10000) / 10000,
          threshold,
          deviation: Math.round(deviation * 10000) / 10000,
          decision: "",
          record,
        });
      }
    }
  }

  /**
   * @param {string} rawLine  one non-empty JSONL line
   * @param {number} lineNo   1-based number of this NON-EMPTY line (matches the
   *                          historical behaviour of cleanJsonlDataset)
   * @param {string} [sourceHint]  input path; used as the source tag when the
   *                          record carries no `source` field
   * @returns {{action: "keep", record: object, source: string} |
   *           {action: "drop", rule: string, line: number}}
   */
  push(rawLine, lineNo, sourceHint = "") {
    this.inputCount += 1;
    let record;
    try {
      record = JSON.parse(rawLine);
    } catch {
      this.#drop("invalid", lineNo, "");
      return { action: "drop", rule: "invalid", line: lineNo };
    }

    const detected = detectFormat(record);
    if (detected === null) {
      this.#drop("invalid", lineNo, "");
      return { action: "drop", rule: "invalid", line: lineNo };
    }
    if (this.format === null) this.format = detected;
    else if (this.format !== detected) {
      // Mixed-format files are invalid for training: keep only the format of
      // the first valid record so the output is a clean single-format file.
      this.#drop("invalid", lineNo, "");
      return { action: "drop", rule: "invalid", line: lineNo };
    }

    const source =
      typeof record.source === "string" && record.source.length > 0 ? record.source : sourceHint;
    const opts = resolveEffectiveOpts(this.baseOpts, source);

    if (opts.cleanText) {
      for (const [owner, key] of textFieldRefs(record, this.format)) {
        owner[key] = cleanTextValue(owner[key], opts);
      }
    }
    if (opts.mergeSameRole && (this.format === "chat" || this.format === "sharegpt")) {
      mergeSameRole(record, this.format);
    }
    if (opts.cleanText) {
      if (requiredFieldRefs(record, this.format).some(([owner, key]) => owner[key].length === 0)) {
        this.#drop("empty_after_clean", lineNo, source);
        return { action: "drop", rule: "empty_after_clean", line: lineNo };
      }
    }

    if (opts.dedupe) {
      const key = canonicalKey(record);
      if (this.seen.has(key)) {
        this.#drop("dedupe", lineNo, source);
        return { action: "drop", rule: "dedupe", line: lineNo };
      }
      this.seen.add(key);
    }

    const texts = textFieldRefs(record, this.format).map(([owner, key]) => owner[key]);
    const totalChars = texts.reduce((sum, value) => sum + value.length, 0);

    // Fuzzy dedupe (second dedupe pass, right after exact dedupe). The hash is
    // only added to the index once the record is actually kept, so a record
    // that later fails a threshold rule cannot poison the index.
    let pendingHash = null;
    if (opts.fuzzyDedupe && opts.simhashDistance > 0) {
      const tokens = simhashTokens(texts.join("\n"));
      if (tokens.length >= opts.simhashMinTokens) {
        pendingHash = simhash64(tokens);
        if (this.simhashIndex.isNearDuplicate(pendingHash)) {
          this.#drop("fuzzy_dupe", lineNo, source);
          return { action: "drop", rule: "fuzzy_dupe", line: lineNo };
        }
      }
    }

    if (opts.minChars > 0 && totalChars < opts.minChars) {
      this.#dropNumeric("min_chars", lineNo, source, totalChars, opts.minChars, record, opts);
      return { action: "drop", rule: "min_chars", line: lineNo };
    }
    if (opts.maxChars > 0 && totalChars > opts.maxChars) {
      this.#dropNumeric("max_chars", lineNo, source, totalChars, opts.maxChars, record, opts);
      return { action: "drop", rule: "max_chars", line: lineNo };
    }
    const repeat = worstRepeatRatio(texts);
    if (opts.repeatRatio > 0 && repeat > opts.repeatRatio) {
      this.#dropNumeric("repeat_ratio", lineNo, source, repeat, opts.repeatRatio, record, opts);
      return { action: "drop", rule: "repeat_ratio", line: lineNo };
    }
    const meaningful = meaningfulRatio(texts);
    if (opts.minMeaningfulRatio > 0 && meaningful < opts.minMeaningfulRatio) {
      this.#dropNumeric("meaningful_ratio", lineNo, source, meaningful, opts.minMeaningfulRatio, record, opts);
      return { action: "drop", rule: "meaningful_ratio", line: lineNo };
    }
    if (opts.requireAssistantEnd && !endsWithAssistant(record, this.format)) {
      this.#drop("ends_assistant", lineNo, source);
      return { action: "drop", rule: "ends_assistant", line: lineNo };
    }

    if (pendingHash !== null) this.simhashIndex.add(pendingHash);
    this.keptCount += 1;
    this.#trackSource(source, true);
    return { action: "keep", record, source };
  }

  get summary() {
    return {
      inputLines: this.inputCount,
      keptLines: this.keptCount,
      removedLines: this.inputCount - this.keptCount,
      format: this.format ?? "empty",
      byRule: this.byRule,
      examples: this.examples,
      bySource: Object.fromEntries(this.bySource),
      // Every kept record was JSON-parsed and format-checked, so a written
      // output is single-format and trainable by construction.
      outputValid: true,
      quarantine: {
        enabled: this.baseOpts.quarantine === true,
        count: this.quarantineRecords.length,
        records: this.quarantineRecords,
      },
    };
  }
}

/**
 * Clean JSONL fine-tuning content (in-memory wrapper over CleanPipeline).
 *
 * @param {string} text raw file content
 * @param {Partial<typeof CLEAN_DEFAULTS>} [options] rule toggles / thresholds
 * @returns {{
 *   inputLines: number,
 *   keptLines: number,
 *   removedLines: number,
 *   format: "chat" | "prompt-completion" | "alpaca" | "sharegpt" | "empty",
 *   byRule: Record<string, number>,
 *   examples: { line: number, rule: string }[],
 *   bySource: Record<string, { count: number, kept: number, removed: number }>,
 *   outputValid: boolean,
 *   quarantine: { enabled: boolean, count: number, records: object[] },
 *   text: string,
 * }}
 */
export function cleanJsonlDataset(text, options = {}) {
  const pipeline = new CleanPipeline(options);
  const kept = [];
  let lineNo = 0;
  for (const line of text.split(/\r?\n/)) {
    if (line.trim() === "") continue;
    lineNo += 1;
    const result = pipeline.push(line, lineNo);
    if (result.action === "keep") kept.push(result.record);
  }
  const summary = pipeline.summary;
  return {
    ...summary,
    text: kept.length > 0 ? `${kept.map((record) => JSON.stringify(record)).join("\n")}\n` : "",
  };
}
