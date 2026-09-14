#!/usr/bin/env python3
"""Layer-2/3 dataset cleaning for dsh-finetune.

Runs AFTER the JS stage-1 pass (lib/clean.js). Reads a stage-1-cleaned JSONL,
applies the enabled checks, and writes the final JSONL plus a JSON report for
the Node side to merge. Same config-file protocol as train.py:

    python clean.py --config /path/to/config.json

Config keys (all optional unless marked):
    data_path      (required) stage-1 cleaned JSONL
    out_path       (required) final JSONL (written atomically: out_path.tmp, then rename)
    report_path    (required) JSON report consumed by lib/tools/dataset-clean.js
    model_path     (required) local HF model dir/id for PPL + embeddings
    source_hint    (optional) original input path, used as the source tag when a
                   record carries no `source` field (mirrors the JS layer)
    device         cpu | cuda | mps | auto (default auto)
    dtype          auto | bf16 | float32 (default auto; bf16 only on CUDA)
    batch_size     default 32
    ppl            bool, default false — drop records whose PPL > ppl_max
    ppl_max        float, default 150.0 (per_source.ppl_max overrides per source).
                   Calibrated on Qwen3-0.6B with >= ppl_min_tokens tokens:
                   p50≈43 / p90≈66 / p95≈80 / p99≈147 on clean Chinese QA.
                   150 is a safe upper bound (only the far tail); tighten per
                   source after reading the report's quantiles and the
                   quarantine boundary samples.
    ppl_min_tokens int, default 32 — records shorter than this many tokens are
                   NOT PPL-scored (kept): short-text PPL is high-variance and
                   would drop clean one-line QA. Short garbage is already
                   handled by stage-1 min_chars/repeat/meaningful rules.
    per_source     [{match, ppl_max}] — first substring match on the record's
                   source tag wins (code/math text has naturally higher PPL)
    lang           bool, default false — drop records whose dominant language is
                   not in allowed_langs (script-ratio detector, no extra deps)
    allowed_langs  list, default ["zh", "en"]
    lang_model     str, default "" — path to a fasttext lid model (e.g.
                   lid.176.bin). When set, the language check uses fasttext
                   predictions instead of the script-share heuristic; falls
                   back to the heuristic automatically if the module or model
                   file is unavailable. Use it when the corpus mixes several
                   Latin-script languages (en/fr/de/es...) that script shares
                   cannot tell apart.
    instruction_follow bool, default false — light non-answer heuristic on the
                   final assistant turn (too-short / question-ending answers)
    semantic_dedup bool, default false — embedding cosine dedup (numpy; no faiss)
    semantic_sim   float, default 0.95
    topic_downsample bool, default false — numpy k-means cap per topic cluster
    topic_max_per_cluster int, default 500
    quarantine_path str — append PPL boundary records (PPL within
                   quarantine_margin of the effective ppl_max) in the same
                   schema as the JS quarantine layer
    quarantine_margin float, default 0.25
    quarantine_max  int, default 5000

Drop order per record (first failing rule wins): lang -> inst -> ppl ->
semantic_dupe -> topic_downsample. The report carries quantiles so thresholds
can be tuned per source (see <path>.clean.python-report.json).
"""

import argparse
import json
import math
import os
import random
import re
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import record_to_messages, detect_format  # noqa: E402

SCRIPT_RE = {
    "zh": re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"),  # Han
    "ja": re.compile(r"[\u3040-\u30ff]"),  # kana
    "ko": re.compile(r"[\uac00-\ud7af]"),  # hangul
    "en": re.compile(r"[A-Za-z]"),
}

# Script-neutral characters that never vote for a language: CJK / full-width /
# typographic punctuation, en/em dash, ellipsis, curly quotes, nbsp.
PUNCT_RE = re.compile(
    r"[\u3000-\u303f\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff5e\ufe30-\ufe4f"
    r"\u2013\u2014\u2018\u2019\u201c\u201d\u2026\u00a0]"
)


def load_config():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = json.load(handle)
    for key in ("model_path", "data_path", "out_path", "report_path"):
        if not cfg.get(key):
            parser.error(f"config key {key!r} is required")
    return cfg


# ---------------------------------------------------------------------------
# Language detection (script-ratio heuristic — no langdetect/fasttext needed)
# ---------------------------------------------------------------------------

def script_shares(text):
    """Share of each script in the text; None when there is no script content.

    Digits, whitespace and punctuation are script-neutral and never vote.
    Latin is counted by WORD, not by character: technical terms such as
    "HTTP" / "HTTPS" / "[EMAIL]" then weigh as one token instead of drowning
    out a Chinese prompt's Han characters.
    """
    import string

    counts = {"zh": 0, "ja": 0, "ko": 0, "en": 0, "other": 0}
    for ch in text:
        if ch.isspace() or ch.isdigit() or ch in string.punctuation:
            continue
        if PUNCT_RE.match(ch):
            continue
        if ch.isascii():
            if ch.isalpha():
                continue  # counted once per word below
            counts["other"] += 1
            continue
        matched = False
        for lang, regex in SCRIPT_RE.items():
            if regex.match(ch):
                counts[lang] += 1
                matched = True
                break
        if not matched:
            counts["other"] += 1
    counts["en"] += len(re.findall(r"[A-Za-z]+", text))
    total = sum(counts.values())
    if total == 0:
        return None
    # Japanese mixes kana + kanji: when kana is present, fold Han into ja.
    if counts["ja"] > 0:
        counts["ja"] += counts["zh"]
        counts["zh"] = 0
    return {lang: count / total for lang, count in counts.items()}


def lang_ok(text, allowed):
    """Keep: dominant script in allowed with >=50% share, or allowed scripts
    together cover >=85% (mixed zh/en code/math text passes either way)."""
    shares = script_shares(text)
    if shares is None:
        return True
    allowed_share = sum(shares.get(lang, 0.0) for lang in allowed)
    dominant = max(shares, key=shares.get)
    return (dominant in allowed and shares[dominant] >= 0.5) or allowed_share >= 0.85


def load_lang_model(cfg):
    """Load the fasttext lid model when cfg.lang_model is set.

    Returns None when the path is empty OR the module/model is unavailable —
    the caller then falls back to the script-share heuristic, so a missing
    optional dependency never fails the run.
    """
    path = cfg.get("lang_model")
    if not path:
        return None
    try:
        import fasttext

        return fasttext.load_model(path)
    except Exception as exc:  # noqa: BLE001 -- optional dependency, per-run tolerance
        print(f"[dsh-clean] WARN lang_model {path!r} unavailable ({exc}); falling back to script-share heuristic", flush=True)
        return None


# ---------------------------------------------------------------------------
# Instruction-following heuristic (light, conservative, toggled off by default)
# ---------------------------------------------------------------------------

def inst_flags(messages):
    if not messages or messages[-1].get("role") != "assistant":
        return ["no_assistant_answer"]
    answer = messages[-1]["content"].strip()
    prompt = "".join(m["content"] for m in messages[:-1]).strip()
    flags = []
    if len(answer) < 4 and len(prompt) > 30:
        flags.append("answer_too_short")
    if answer.endswith(("？", "?")) and len(answer) < 50:
        flags.append("answer_is_question")
    if prompt and len(prompt) > 50 and len(answer) / len(prompt) < 0.03:
        flags.append("answer_ratio_too_low")
    return flags


# ---------------------------------------------------------------------------
# Model: PPL + embeddings
# ---------------------------------------------------------------------------

def load_model(cfg):
    import torch

    device_cfg = cfg.get("device", "auto")
    if device_cfg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_cfg
    dtype_cfg = (os.environ.get("DSH_FINETUNE_DTYPE") or cfg.get("dtype") or "auto").strip().lower()
    torch_dtype = torch.bfloat16 if dtype_cfg in ("auto", "bf16") and device == "cuda" else torch.float32

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[dsh-clean] loading tokenizer/model from {cfg['model_path']} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_path"], torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    model.config.use_cache = False
    model.eval()
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[dsh-clean] model loaded: {total_params / 1e6:.0f}M params on {device}", flush=True)
    return tokenizer, model, device


def _make_batch(ids_list, pad_id, device, torch):
    max_len = max(len(ids) for ids in ids_list)
    input_ids = []
    attention_mask = []
    for ids in ids_list:
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
    )


def compute_ppl(model, tokenizer, device, ids_list, batch_size):
    """Per-record perplexity (causal: logits[t] vs input_ids[t+1], pad masked)."""
    import torch
    import torch.nn.functional as F

    results = [None] * len(ids_list)
    for start in range(0, len(ids_list), batch_size):
        batch = ids_list[start : start + batch_size]
        input_ids, attention_mask = _make_batch(batch, tokenizer.pad_token_id, device, torch)
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits
        vocab = logits.shape[-1]
        shift_logits = logits[:, :-1, :].reshape(-1, vocab).float()
        shift_labels = input_ids[:, 1:].reshape(-1)
        shift_mask = attention_mask[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, reduction="none")
        loss = (loss * shift_mask).reshape(len(batch), -1)
        denom = shift_mask.reshape(len(batch), -1).sum(dim=1).clamp(min=1)
        ppl_per_row = (loss.sum(dim=1) / denom).tolist()
        for offset, ppl in enumerate(ppl_per_row):
            results[start + offset] = math.exp(ppl)
    return results


def compute_embeddings(model, tokenizer, device, ids_list, batch_size):
    """Mean-pooled last hidden state over non-pad tokens -> numpy (n, d)."""
    import numpy as np
    import torch

    outs = []
    for start in range(0, len(ids_list), batch_size):
        batch = ids_list[start : start + batch_size]
        input_ids, attention_mask = _make_batch(batch, tokenizer.pad_token_id, device, torch)
        with torch.no_grad():
            hidden = model(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1].float()
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        outs.append(pooled.cpu().numpy())
    return np.concatenate(outs, axis=0)


def kmeans_cosine(X, k, iters=20, seed=42):
    """Cosine k-means on L2-normalized X; returns cluster labels (n,)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = min(k, n)
    centroids = X[rng.choice(n, size=k, replace=False)].copy()
    for _ in range(iters):
        labels = (X @ centroids.T).argmax(axis=1)
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k)
        for c in range(k):
            members = X[labels == c]
            if len(members):
                new_centroids[c] = members.mean(axis=0)
                counts[c] = len(members)
        empty = np.where(counts == 0)[0]
        if len(empty):
            new_centroids[empty] = X[rng.choice(n, size=len(empty), replace=False)]
        centroids = new_centroids
    return (X @ centroids.T).argmax(axis=1)


def quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    index = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[index]


def effective_ppl_max(source, cfg):
    value = float(cfg.get("ppl_max", 150.0))
    for entry in cfg.get("per_source", []) or []:
        match = entry.get("match")
        if match and source and match.lower() in str(source).lower():
            return float(entry.get("ppl_max", value))
    return value


def main():
    cfg = load_config()
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    import numpy as np

    fmt = None
    records = []
    with open(cfg["data_path"], encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            records.append({"line": line_no, "record": record})
    total = len(records)
    print(f"[dsh-clean] input: {total} records", flush=True)

    dropped = {"ppl": 0, "lang": 0, "inst": 0, "semantic_dupe": 0, "topic_downsample": 0}
    examples = []
    skipped_tokens = 0
    skipped_ppl_short = 0
    kept_mask = [True] * total
    dropped_rule = [None] * total
    ppl_rows = []  # (line, ppl, source, ppl_max, effective_max)
    quarantine = []

    def mark_drop(index, rule, metric=None, source=None):
        if not kept_mask[index]:
            return
        kept_mask[index] = False
        dropped_rule[index] = rule
        dropped[rule] += 1
        if len(examples) < 10:
            example = {"line": records[index]["line"], "rule": rule}
            if isinstance(metric, (int, float)) and not isinstance(metric, bool):
                example["metric"] = round(float(metric), 4)
            if source:
                example["source"] = source
            examples.append(example)

    def source_of(index):
        record = records[index]["record"]
        if isinstance(record.get("source"), str) and record["source"]:
            return record["source"]
        return cfg.get("source_hint") or ""

    # ---- messages / text per record ----
    messages_list = []
    texts = []
    for index, item in enumerate(records):
        detected = detect_format(item["record"])
        if detected is None:
            continue
        if fmt is None:
            fmt = detected
        messages = record_to_messages(item["record"], detected)
        messages_list.append(messages)
        texts.append("\n".join(m["content"] for m in messages))

    # ---- language check ----
    # Evaluated on the PROMPT side (user/system turns) as the primary signal,
    # falling back to the full text when the prompt has little script content.
    # Rationale: the instruction's language decides what the sample trains;
    # a Chinese prompt with a code/English assistant answer is expected (code
    # and math data), so the assistant side must not veto the record.
    # Backend: fasttext when cfg.lang_model is set (falls back to the
    # script-share heuristic if the module or model file is missing).
    if cfg.get("lang"):
        allowed = cfg.get("allowed_langs") or ["zh", "en"]
        lang_model = load_lang_model(cfg)
        for index in range(total):
            if not kept_mask[index] or index >= len(messages_list):
                continue
            messages = messages_list[index]
            prompt_text = "".join(m["content"] for m in messages if m["role"] in ("user", "system"))
            full_text = texts[index]
            if lang_model is not None:
                probe = (prompt_text if prompt_text else full_text).replace("\n", " ")
                if probe.strip():
                    preds = lang_model.predict(probe, k=1)
                    code = preds[0][0].replace("__label__", "")
                    if code not in allowed:
                        mark_drop(index, "lang", metric=round(float(preds[1][0]), 4), source=source_of(index))
            else:
                probe = prompt_text if script_shares(prompt_text) else full_text
                if not lang_ok(probe, allowed):
                    shares = script_shares(probe) or {}
                    dominant = max(shares, key=shares.get) if shares else "?"
                    mark_drop(index, "lang", metric=round(shares.get(dominant, 0.0), 4), source=source_of(index))

    # ---- instruction-following heuristic ----
    if cfg.get("instruction_follow"):
        for index in range(total):
            if not kept_mask[index] or index >= len(messages_list):
                continue
            flags = inst_flags(messages_list[index])
            if flags:
                mark_drop(index, "inst", source=source_of(index))

    # ---- PPL / embeddings (model-backed) ----
    needs_model = cfg.get("ppl") or cfg.get("semantic_dedup") or cfg.get("topic_downsample")
    ids_list = []
    ids_index = []  # global record index per ids_list slot
    if needs_model:
        tokenizer, model, device = load_model(cfg)
        batch_size = int(cfg.get("batch_size", 32))
        for index in range(total):
            if not kept_mask[index] or index >= len(messages_list):
                continue
            try:
                ids = tokenizer.apply_chat_template(messages_list[index], tokenize=True, add_generation_prompt=False)
            except Exception as exc:  # noqa: BLE001 -- per-sample tolerance
                print(f"[dsh-clean] WARN record {records[index]['line']}: tokenisation failed: {exc}", flush=True)
                skipped_tokens += 1
                continue
            if not ids:
                skipped_tokens += 1
                continue
            ids_list.append(ids)
            ids_index.append(index)

        if cfg.get("ppl"):
            ppl_min_tokens = int(cfg.get("ppl_min_tokens", 32))
            ppl_scores = compute_ppl(model, tokenizer, device, ids_list, batch_size)
            for slot, ppl in enumerate(ppl_scores):
                if ppl is None:
                    continue
                index = ids_index[slot]
                if len(ids_list[slot]) < ppl_min_tokens:
                    # Short-text PPL is high-variance; short garbage is already
                    # handled by the stage-1 min_chars/repeat/meaningful rules.
                    skipped_ppl_short += 1
                    continue
                source = source_of(index)
                threshold = effective_ppl_max(source, cfg)
                ppl_rows.append((records[index]["line"], ppl, source, threshold))
                if ppl > threshold:
                    mark_drop(index, "ppl", metric=ppl, source=source)
                    margin = float(cfg.get("quarantine_margin", 0.25))
                    if (
                        cfg.get("quarantine_path")
                        and threshold > 0
                        and len(quarantine) < int(cfg.get("quarantine_max", 5000))
                    ):
                        deviation = (ppl - threshold) / threshold
                        if 0 < deviation <= margin:
                            quarantine.append(
                                {
                                    "line": records[index]["line"],
                                    "rule": "ppl",
                                    "source": source or None,
                                    "metric": round(ppl, 4),
                                    "threshold": threshold,
                                    "deviation": round(deviation, 4),
                                    "decision": "",
                                    "record": records[index]["record"],
                                }
                            )

        if cfg.get("semantic_dedup") or cfg.get("topic_downsample"):
            survivors = [slot for slot, index in enumerate(ids_index) if kept_mask[index]]
            if survivors:
                embeddings = compute_embeddings(model, tokenizer, device, [ids_list[s] for s in survivors], batch_size)
                norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)
                # norm[position] aligns with survivors[position]

                # semantic dedup (cosine against all earlier kept embeddings)
                keep_positions = list(range(len(survivors)))
                if cfg.get("semantic_dedup"):
                    sim_threshold = float(cfg.get("semantic_sim", 0.95))
                    kept_embeddings = []
                    filtered = []
                    for position, slot in enumerate(survivors):
                        index = ids_index[slot]
                        sim = 0.0
                        if kept_embeddings:
                            sim = float(np.max(np.asarray(kept_embeddings, dtype=np.float32) @ norm[position]))
                        if kept_embeddings and sim >= sim_threshold:
                            mark_drop(index, "semantic_dupe", metric=round(sim, 4), source=source_of(index))
                        else:
                            kept_embeddings.append(norm[position])
                            filtered.append(position)
                    keep_positions = filtered

                # topic downsampling (k-means cap per cluster)
                if cfg.get("topic_downsample") and len(keep_positions) > 1:
                    cap = int(cfg.get("topic_max_per_cluster", 500))
                    sub = np.asarray([norm[p] for p in keep_positions], dtype=np.float32)
                    k = min(64, max(4, int(math.sqrt(len(keep_positions)) / 2)))
                    k = min(k, len(keep_positions))
                    labels = kmeans_cosine(sub, k, seed=seed)
                    counts = {}
                    for label in labels:
                        counts[label] = counts.get(label, 0) + 1
                    seen = {cluster: 0 for cluster in counts}
                    for position, label in enumerate(labels):
                        if counts[label] <= cap:
                            continue
                        if seen[label] < cap:
                            seen[label] += 1
                        else:
                            slot = survivors[keep_positions[position]]
                            mark_drop(ids_index[slot], "topic_downsample", source=source_of(ids_index[slot]))

    # ---- write final JSONL (atomic) + report ----
    kept_lines = 0
    with open(cfg["out_path"] + ".tmp", "w", encoding="utf-8") as handle:
        for index, item in enumerate(records):
            if kept_mask[index]:
                handle.write(json.dumps(item["record"], ensure_ascii=False) + "\n")
                kept_lines += 1
    os.replace(cfg["out_path"] + ".tmp", cfg["out_path"])

    if quarantine:
        with open(cfg["quarantine_path"], "a", encoding="utf-8") as handle:
            for entry in quarantine:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    ppl_scores = [row[1] for row in ppl_rows]
    ppl_sorted = sorted(ppl_scores)
    by_source = {}
    for line, ppl, source, threshold in ppl_rows:
        entry = by_source.setdefault(source or "(none)", {"count": 0, "ppl_sum": 0.0, "ppl_list": [], "dropped_ppl": 0})
        entry["count"] += 1
        entry["ppl_sum"] += ppl
        entry["ppl_list"].append(ppl)
    by_source_out = {}
    for source, entry in by_source.items():
        sorted_list = sorted(entry["ppl_list"])
        by_source_out[source] = {
            "count": entry["count"],
            "mean_ppl": round(entry["ppl_sum"] / entry["count"], 2),
            "p90_ppl": round(quantile(sorted_list, 0.9) or 0.0, 2),
            "dropped_ppl": sum(1 for row in ppl_rows if row[2] == source and row[1] > row[3]),
        }

    report = {
        "total": total,
        "kept": kept_lines,
        "dropped": dropped,
        "format": fmt,
        "skipped_tokenization": skipped_tokens,
        "skipped_ppl_short": skipped_ppl_short,
        "ppl": (
            {
                "count": len(ppl_scores),
                "min": round(ppl_sorted[0], 2) if ppl_sorted else None,
                "mean": round(sum(ppl_scores) / len(ppl_scores), 2) if ppl_scores else None,
                "p50": round(quantile(ppl_sorted, 0.5) or 0.0, 2) if ppl_sorted else None,
                "p90": round(quantile(ppl_sorted, 0.9) or 0.0, 2) if ppl_sorted else None,
                "p95": round(quantile(ppl_sorted, 0.95) or 0.0, 2) if ppl_sorted else None,
                "p99": round(quantile(ppl_sorted, 0.99) or 0.0, 2) if ppl_sorted else None,
                "max": round(ppl_sorted[-1], 2) if ppl_sorted else None,
            }
            if ppl_scores
            else {}
        ),
        "by_source": by_source_out,
        "examples": examples,
        "suggested_ppl_max": round(quantile(ppl_sorted, 0.9) or 0.0, 2) if ppl_sorted else None,
        "skipped": [] if needs_model else ["model not loaded (no ppl/semantic/topic flag enabled)"],
    }
    with open(cfg["report_path"], "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(
        f"[dsh-clean] done: {total} -> {kept_lines} records; dropped {json.dumps(dropped, ensure_ascii=False)}; "
        f"report at {cfg['report_path']}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 -- surface a clean nonzero exit
        print(f"[dsh-clean] FATAL: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
