#!/usr/bin/env python3
"""Minimal self-contained LoRA SFT trainer for dsh-finetune.

Deliberately does NOT import `trl`, or `datasets`. It depends
only on `torch` + `transformers` + `peft` and drives a standard
`transformers.Trainer` over a small in-memory `torch.utils.data.Dataset`.

Input:  a JSONL file whose records are any of chat / prompt-completion /
        alpaca / sharegpt (matching `lib/dataset.js` in the Node plugin).
Output: a PEFT LoRA adapter + tokenizer saved to `output_dir`.

Usage:
    python train.py --config /path/to/config.json

Config JSON keys (all optional unless marked):
    model_path  (required) local dir or HF model id (offline HF cache is used)
    data_path   (required) path to the JSONL dataset
    output_dir  (required) where the adapter is saved
    format      auto | chat | prompt-completion | alpaca | sharegpt (default auto)
    device      cpu | cuda (default cpu; cuda only when available)
    dtype       auto | bf16 | float32 (default auto -> bfloat16 when supported);
                override via env DSH_FINETUNE_DTYPE, e.g. "float32" on CPUs
                without native bf16 support where bf16 GEMM is very slow
    lora_r              default 8
    lora_alpha          default 16
    lora_dropout        default 0.05
    learning_rate       default 2e-4
    num_epochs          default 3.0
    max_steps           default 0 (0 = train until num_epochs)
    batch_size          default 2
    grad_accum          default 1
    max_seq_len         default 1024 (overlong samples are skipped)
    logging_steps       default 10
    save_steps          default 200
    save_total_limit    default 1
    seed                default 42
"""

import argparse
import json
import os
import random
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

def load_config():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = json.load(handle)
    for key in ("model_path", "data_path", "output_dir"):
        if not cfg.get(key):
            parser.error(f"config key {key!r} is required")
    return cfg

# --------------------------------------------------------------------------
# Dataset: any supported record -> list of chat messages -> tokenized rows
# --------------------------------------------------------------------------

def record_to_messages(record, fmt):
    """Normalise one JSONL record into [{role, content}, ...]."""
    if fmt == "chat":
        return record["messages"]
    if fmt == "prompt-completion":
        return [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["completion"]},
        ]
    if fmt == "alpaca":
        text = record["instruction"]
        if isinstance(record.get("input"), str) and record["input"].strip():
            text = f"{text}\n\n{record['input']}"
        return [
            {"role": "user", "content": text},
            {"role": "assistant", "content": record["output"]},
        ]
    if fmt == "sharegpt":
        role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant", "system": "system"}
        messages = []
        for turn in record["conversations"]:
            role = role_map.get(turn["from"])
            if role == "system":
                if messages and messages[0]["role"] == "system":
                    continue  # one system turn per sample is enough
                messages.insert(0, {"role": "system", "content": turn["value"]})
            else:
                messages.append({"role": role, "content": turn["value"]})
        return messages
    raise ValueError(f"unknown format {fmt}")

def detect_format(record):
    if isinstance(record.get("messages"), list) and record["messages"]:
        return "chat"
    if "prompt" in record and "completion" in record:
        return "prompt-completion"
    if "instruction" in record and "output" in record:
        return "alpaca"
    if "conversations" in record:
        return "sharegpt"
    return None

def main():
    cfg = load_config()
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    import torch
    from torch.utils.data import Dataset

    device = cfg.get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = None
    model = None
    # Loaded below after the transformers/peft imports, so import failures
    # stay readable and reach the FATAL handler cleanly.

    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    print(f"[dsh-finetune] loading tokenizer/model from {cfg['model_path']} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # dtype resolution: env DSH_FINETUNE_DTYPE wins, then config "dtype", then auto.
    # "auto" is device-aware: bf16 only where the hardware can actually run it,
    # because bf16 GEMM is pathologically slow on CPUs without native support
    # (e.g. Broadwell-era Intel: AVX2 only, no AVX512-BF16/AMX) — so auto
    # resolves to float32 on CPU and bf16 on CUDA. Pass "bf16" explicitly to
    # force bf16 on a CPU anyway.
    dtype_cfg = (os.environ.get("DSH_FINETUNE_DTYPE") or cfg.get("dtype") or "auto").strip().lower()
    if dtype_cfg == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype_cfg == "auto":
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        torch_dtype = torch.float32
    print(f"[dsh-finetune] dtype: requested={dtype_cfg} resolved={torch_dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_path"], torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    model.config.use_cache = False
    if device == "cpu":
        model = model.to("cpu")
    elif torch.cuda.is_available():
        model = model.to("cuda")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[dsh-finetune] model loaded: {total_params / 1e6:.0f}M params on {device}", flush=True)

    # ---- Tokenise the dataset to (input_ids, labels) with assistant masking ----
    #
    # Qwen3's chat template has no `{% generation %}` marker, so
    # `return_assistant_tokens_mask=True` yields an all-zero mask. Instead we
    # locate the final assistant answer by tokenising the prompt separately:
    #   prompt = template(messages[:-1], add_generation_prompt=True)   # ends w/ assistant header
    #   full   = template(messages)                                    # prompt + final answer
    # and require full[:len(prompt)] == prompt before masking the tail.
    # Multi-turn assistant blocks before the final answer stay in the prompt
    # region (unmasked) — a known simplification for minimal trainers.
    max_seq_len = int(cfg.get("max_seq_len", 1024))
    fmt_override = None
    tokenized = []
    with open(cfg["data_path"], encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[dsh-finetune] WARN line {line_no}: {exc}", flush=True)
                continue
            fmt = fmt_override or detect_format(record)
            if fmt is None:
                print(f"[dsh-finetune] WARN line {line_no}: unrecognised record", flush=True)
                continue
            fmt_override = fmt
            messages = record_to_messages(record, fmt)
            if not messages or messages[-1].get("role") != "assistant":
                print(f"[dsh-finetune] WARN line {line_no}: sample does not end on an assistant answer; skipped", flush=True)
                continue
            try:
                full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
                prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
            except Exception as exc:  # noqa: BLE001 -- per-sample tolerance
                print(f"[dsh-finetune] WARN line {line_no}: tokenisation failed: {exc}", flush=True)
                continue
            if full[: len(prompt)] != prompt:
                # Template did not align as assumed: fall back to a fully
                # supervised sample rather than silently mislabelling.
                print(f"[dsh-finetune] WARN line {line_no}: prompt/full alignment failed; training on whole sample", flush=True)
                labels = list(full)
            else:
                labels = [-100] * len(prompt) + full[len(prompt):]
            if len(full) > max_seq_len:
                # Truncate from the right; drop the sample if it loses its target.
                if sum(1 for token_id in labels[:max_seq_len] if token_id != -100) == 0:
                    print(f"[dsh-finetune] WARN line {line_no}: truncated away the assistant target", flush=True)
                    continue
                full = full[:max_seq_len]
                labels = labels[:max_seq_len]
            tokenized.append({"input_ids": full, "labels": labels})

    if not tokenized:
        raise RuntimeError("no usable samples after tokenisation")
    print(f"[dsh-finetune] dataset: {len(tokenized)} samples, format={fmt_override}", flush=True)

    class TrainSet(Dataset):
        def __len__(self):
            return len(tokenized)

        def __getitem__(self, index):
            return tokenized[index]

    def collate(batch):
        pad = tokenizer.pad_token_id
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids, labels = [], []
        for item in batch:
            length = len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad] * (max_len - length))
            labels.append(item["labels"] + [-100] * (max_len - length))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": (torch.tensor(input_ids, dtype=torch.long) != pad).long(),
        }

    # ---- LoRA ----
    from peft import LoraConfig, get_peft_model, TaskType

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(cfg.get("lora_r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ---- Train ----
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    max_steps = int(cfg.get("max_steps", 0))
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(cfg.get("batch_size", 2)),
        gradient_accumulation_steps=int(cfg.get("grad_accum", 1)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        num_train_epochs=float(cfg.get("num_epochs", 3.0)),
        max_steps=max_steps if max_steps > 0 else -1,
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_strategy="steps",
        save_steps=int(cfg.get("save_steps", 200)),
        save_total_limit=int(cfg.get("save_total_limit", 1)),
        log_level="warning",
        report_to=[],
        seed=seed,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        no_cuda=(device != "cuda"),
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TrainSet(),
        data_collator=collate,
        tokenizer=tokenizer,
    )
    print("[dsh-finetune] training starts", flush=True)
    result = trainer.train()
    print(f"[dsh-finetune] training finished; loss ~ {result.training_loss:.4f}", flush=True)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "training_state.json"), "w", encoding="utf-8") as handle:
        json.dump({"training_loss": result.training_loss, "steps": result.global_step}, handle, indent=2)
    print(f"[dsh-finetune] adapter saved to {output_dir}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 -- surface a clean nonzero exit
        print(f"[dsh-finetune] FATAL: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
