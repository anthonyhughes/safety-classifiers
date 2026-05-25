#!/usr/bin/env python3
"""
Phase 2 — Fine-tune classifiers with LoRA.

Trains binary sequence classifiers on top of Llama models using LoRA
adapters + a fully-trained classification head.  The classification task
is automatically detected from the prepared data:

Usage:
    python finetune.py --model_size 1b --split A
    python finetune.py --model_size all --split all
    python finetune.py --model_size 8b --split all --quantize
    python finetune.py --model_size 1b --split A --dry_run
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import config
from config import set_seed
from utils import (
    compute_classification_metrics,
    format_for_classification,
    format_multiturn_for_classification,
    load_tokenizer,
    save_json,
    set_plot_style,
)


def str2bool(v):
    """Parse boolean from string for argparse compatibility."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


# =========================================================================== #
# Tokenization
# =========================================================================== #

def tokenize_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_length: int = config.MAX_SEQ_LEN,
    task: str = "safety",
    multiturn: bool = False,
    label_mode: str = "binary",
) -> Dataset:
    """Tokenize text column and create a HuggingFace Dataset.

    When *multiturn* is ``True`` the ``conversations`` column (JSON-
    encoded list of turn dicts) is used instead of ``text``, and each
    conversation is formatted via :func:`format_multiturn_for_classification`.
    """
    if multiturn and "conversations" in df.columns:
        import json as _json
        texts = []
        for conv_json in df["conversations"].tolist():
            turns = _json.loads(conv_json)
            texts.append(format_multiturn_for_classification(turns, tokenizer, task=task))
    else:
        texts = [format_for_classification(t, tokenizer, task=task,
                                           label_mode=label_mode)
                 for t in df["text"].tolist()]

    ds = Dataset.from_dict({
        "text": texts,
        "label": df["label"].tolist(),
    })

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # dynamic padding via DataCollator
            max_length=max_length,
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    return ds


# =========================================================================== #
# Model loading
# =========================================================================== #

def load_model_and_tokenizer(
    model_size: str,
    quantize: bool = False,
    num_labels: int = 2,
):
    """Load base model + tokenizer, optionally with 4-bit quantization."""
    model_name = config.MODELS[model_size]
    tokenizer = load_tokenizer(model_name)

    load_kwargs: dict = {
        "num_labels": num_labels,
        "torch_dtype": torch.bfloat16,
    }

    # Detect distributed mode: check multiple env vars set by torchrun/accelerate
    is_distributed = (
        int(os.environ.get("WORLD_SIZE", "1")) > 1
        or os.environ.get("LOCAL_RANK") is not None
        or os.environ.get("RANK") is not None
    )

    if quantize:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
    elif is_distributed:
        # DeepSpeed / multi-GPU: let the framework handle device placement
        pass
    else:
        load_kwargs["device_map"] = "auto"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, **load_kwargs
    )

    # Set pad token on model config
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def apply_lora(
    model,
    quantize: bool = False,
    lora_r: int | None = None,
    lora_alpha: int | None = None,
    lora_dropout: float | None = None,
    lora_target_modules: list[str] | None = None,
):
    """Wrap model with LoRA adapter; classification head is auto-saved."""
    from peft import LoraConfig, TaskType, get_peft_model

    if quantize:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    r = lora_r if lora_r is not None else config.LORA_R
    alpha = lora_alpha if lora_alpha is not None else config.LORA_ALPHA
    dropout = lora_dropout if lora_dropout is not None else config.LORA_DROPOUT
    targets = lora_target_modules if lora_target_modules is not None else config.LORA_TARGET_MODULES

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=targets,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,  # auto-includes score head in modules_to_save
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# =========================================================================== #
# Custom metrics for Trainer
# =========================================================================== #

def make_compute_metrics_fn(num_labels: int = 2):
    """Return a compute_metrics callable for the HF Trainer."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = np.array(logits)
        labels = np.array(labels)
        preds = np.argmax(logits, axis=-1)

        if num_labels <= 2:
            probs = _softmax(logits)[:, 1]
            metrics = compute_classification_metrics(preds, labels, probs)
        else:
            # Multi-class: use true-class probability for AUC
            probs = _softmax(logits)
            metrics = compute_classification_metrics(
                preds, labels, probs, num_labels=num_labels,
            )
        return metrics

    return compute_metrics


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# =========================================================================== #
# Training curves
# =========================================================================== #

def plot_training_curves(log_history: list[dict], save_path: Path) -> None:
    """Plot loss and AUC curves from Trainer log history."""
    set_plot_style()

    train_steps, train_loss = [], []
    eval_steps, eval_loss, eval_auc = [], [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry.get("step", 0))
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", 0))
            eval_loss.append(entry["eval_loss"])
            if "eval_auc" in entry:
                eval_auc.append(entry["eval_auc"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax = axes[0]
    if train_steps:
        ax.plot(train_steps, train_loss, label="Train loss", alpha=0.7)
    if eval_steps:
        ax.plot(eval_steps, eval_loss, label="Val loss", marker="o", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()

    # AUC
    ax = axes[1]
    if eval_auc:
        ax.plot(eval_steps[: len(eval_auc)], eval_auc, marker="o", markersize=4, color="green")
    ax.set_xlabel("Step")
    ax.set_ylabel("AUC")
    ax.set_title("Validation AUC")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"  Training curves saved to {save_path}")


# =========================================================================== #
# Single classifier training
# =========================================================================== #

def train_one_classifier(
    model_size: str,
    split: str,
    args: argparse.Namespace,
) -> dict:
    """Train a single classifier and return validation metrics."""
    classifier_name = f"{model_size}_{split}"
    print("\n" + "=" * 70)
    print(f"TRAINING: {classifier_name}")
    print(f"  Model: {config.MODELS[model_size]}")
    print(f"  Training split: {split}_train")
    print(f"  LoRA: {'enabled' if args.use_lora == True else 'disabled'}")
    print("=" * 70)

    set_seed(args.seed)

    # ----- Load data -----
    data_path = Path(args.data_dir) / config.DATA_FILE
    df = pd.read_parquet(data_path)

    # Detect label_mode and dataset_name from metadata or CLI
    label_mode = getattr(args, "label_mode", "binary")
    meta_path = Path(args.data_dir) / "metadata.json"
    dataset_name: str | None = None
    if meta_path.exists():
        import json as _json
        with open(meta_path) as _mf:
            meta = _json.load(_mf)
            dataset_name = meta.get("dataset")
            # Auto-detect label_mode from metadata if not explicitly set
            if label_mode == "binary" and meta.get("label_mode") == "multiclass":
                label_mode = "multiclass"
                print(f"  Label mode from metadata: {label_mode}")

    # Auto-detect task from data
    task = getattr(args, "task", "auto")
    if task == "auto":
        if dataset_name == "pooled":
            task = "pooled"
        elif "language" in df.columns:
            task = "language"
        else:
            task = "safety"
    print(f"  Task: {task}")

    num_labels = config.get_num_labels(dataset_name, label_mode)
    print(f"  Label mode: {label_mode} (num_labels={num_labels})")

    # Detect multi-turn mode (xguard-multiturn) from data
    multiturn = "conversations" in df.columns
    if multiturn:
        print("  Multi-turn mode: enabled (conversations column detected)")

    train_df = df[df["split"] == f"{split}_train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    print(f"  Train size: {len(train_df)} (pos={train_df['label'].sum()}, neg={len(train_df) - train_df['label'].sum()})")
    print(f"  Val size:   {len(val_df)} (pos={val_df['label'].sum()}, neg={len(val_df) - val_df['label'].sum()})")

    # ----- Load model -----
    model, tokenizer = load_model_and_tokenizer(
        model_size, quantize=args.quantize, num_labels=num_labels,
    )
    if args.use_lora:
        model = apply_lora(
            model,
            quantize=args.quantize,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
        )
    else:
        # Full fine-tuning: all parameters remain trainable (no LoRA, no freezing)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  LoRA disabled — full fine-tuning: {trainable:,} / {total:,} params ({100 * trainable / total:.2f}%)")

    # ----- Tokenize -----
    max_length = args.max_seq_len
    train_ds = tokenize_dataset(train_df, tokenizer, max_length, task=task,
                                multiturn=multiturn, label_mode=label_mode)
    val_ds = tokenize_dataset(val_df, tokenizer, max_length, task=task,
                              multiturn=multiturn, label_mode=label_mode)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # ----- Training arguments -----
    ckpt_dir = config.checkpoint_path(model_size, split)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Use a lower LR for full fine-tuning to avoid instability.
    # If the user didn't override --lr, pick from the per-size tables.
    effective_lr = args.lr
    if args.lr == config.LEARNING_RATE:  # user didn't override
        if args.use_lora:
            effective_lr = config.LORA_LR_BY_SIZE.get(model_size, config.LEARNING_RATE)
        else:
            effective_lr = config.FULL_FT_LR_BY_SIZE.get(model_size, config.FULL_FT_LEARNING_RATE)
        if effective_lr != args.lr:
            regime = "LoRA" if args.use_lora else "Full FT"
            print(f"  Auto-adjusted LR for {regime} {model_size}: {config.LEARNING_RATE} → {effective_lr}")

    # Adjust batch sizes for long-context to avoid OOM
    effective_batch_size = args.batch_size
    effective_grad_accum = config.GRADIENT_ACCUMULATION_STEPS
    if max_length > config.MAX_SEQ_LEN:
        # Scale down batch size proportionally, keep effective batch ~32
        scale_factor = max(1, max_length // config.MAX_SEQ_LEN)
        effective_batch_size = max(1, args.batch_size // scale_factor)
        effective_grad_accum = max(1, 32 // effective_batch_size)
        print(f"  Long-context adjustment: batch_size={effective_batch_size}, "
              f"grad_accum={effective_grad_accum} "
              f"(effective={effective_batch_size * effective_grad_accum})")

    # Enable gradient checkpointing for long-context to save memory
    use_grad_ckpt = args.gradient_checkpointing or max_length > config.MAX_SEQ_LEN
    if use_grad_ckpt:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: enabled")

    # Detect DeepSpeed mode for adjusting save/load strategy
    is_deepspeed = (
        "ACCELERATE_LAUNCH" in os.environ
        or os.environ.get("DEEPSPEED_CONFIG_FILE")
        or os.environ.get("LOCAL_RANK") is not None
    )

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=effective_batch_size,
        per_device_eval_batch_size=max(1, effective_batch_size * 2),
        gradient_accumulation_steps=effective_grad_accum,
        learning_rate=effective_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=config.WARMUP_RATIO,
        bf16=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        save_only_model=True,  # Always save just model weights (avoids OOM on best-model reload)
        load_best_model_at_end=not is_deepspeed,  # Disabled for DeepSpeed: reloading ZeRO checkpoint OOMs
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        logging_dir=str(ckpt_dir / "logs"),
        report_to="none",
        remove_unused_columns=False,
        max_steps=10 if args.dry_run else -1,
        seed=args.seed,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
    )

    # ----- Trainer -----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics_fn(num_labels=num_labels),
    )

    # ----- Train -----
    print("\n  Starting training…")
    trainer.train()

    # ----- Reload best model for DeepSpeed -----
    # With DeepSpeed, load_best_model_at_end is disabled to avoid OOM when
    # reloading the full ZeRO optimizer state.  Instead we reload just the
    # model weights from the best checkpoint (saved via save_only_model=True).
    if is_deepspeed and trainer.state.best_model_checkpoint:
        best_path = Path(trainer.state.best_model_checkpoint)
        print(f"  [DeepSpeed] Reloading best model weights from {best_path}")
        from safetensors.torch import load_file as _load_safetensors
        state_dict = {}
        safetensor_files = list(best_path.glob("*.safetensors"))
        if safetensor_files:
            for sf in safetensor_files:
                state_dict.update(_load_safetensors(str(sf), device="cpu"))
        else:
            # Fallback to pytorch bin
            bin_file = best_path / "pytorch_model.bin"
            if bin_file.exists():
                state_dict = torch.load(str(bin_file), map_location="cpu")
        if state_dict:
            unwrapped = trainer.accelerator.unwrap_model(trainer.model)
            unwrapped.load_state_dict(state_dict, strict=False)
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  [DeepSpeed] Best model loaded (epoch {trainer.state.best_metric:.4f} eval_loss)")

    # ----- Evaluate -----
    print("\n  Running final evaluation…")
    eval_results = trainer.evaluate()
    print(f"  Validation results: {json.dumps(eval_results, indent=2)}")

    # ----- Save -----
    if not args.dry_run:
        print(f"\n  Saving checkpoint to {ckpt_dir}")
        trainer.save_model(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))

        # If not using LoRA, remove any stale adapter files from previous runs
        if not args.use_lora:
            for old_file in ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]:
                old_path = Path(str(ckpt_dir)) / old_file
                if old_path.exists():
                    old_path.unlink()
                    print(f"  Removed stale LoRA artifact: {old_file}")

        # Save training mode marker so inference knows how to load
        save_json({"use_lora": args.use_lora}, Path(str(ckpt_dir)) / "training_mode.json")

        # Save metrics
        save_json(eval_results, ckpt_dir / "eval_results.json")
        save_json(trainer.state.log_history, ckpt_dir / "log_history.json")

        # Also save eval_results to scores dir (survives checkpoint deletion)
        save_json(eval_results, config.SCORES_DIR / f"eval_results_{classifier_name}.json")

        # Plot training curves
        plot_training_curves(
            trainer.state.log_history,
            config.RESULTS_DIR / f"training_curves_{classifier_name}.png",
        )
    else:
        print("  (dry run — skipping checkpoint save)")

    # ----- Cleanup -----
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return eval_results


# =========================================================================== #
# CLI
# =========================================================================== #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2: Fine-tune safety classifiers with LoRA."
    )
    p.add_argument("--model_size", choices=list(config.MODELS.keys()) + ["all"], default="all")
    p.add_argument("--split", choices=["A", "B", "all"], default="all")
    p.add_argument("--quantize", action="store_true", help="Use 4-bit quantization (recommended for 8B).")
    p.add_argument("--use_lora", type=str2bool, default=True,
                   help="Use LoRA adapters (default: true). Pass 'false' for full fine-tuning.")
    p.add_argument(
        "--task",
        choices=["safety", "language", "auto"],
        default="auto",
        help="Classification task. 'auto' detects from data (default).",
    )
    p.add_argument("--lora_r", type=int, default=config.LORA_R,
                   help="LoRA rank (default: %(default)s).")
    p.add_argument("--lora_alpha", type=int, default=config.LORA_ALPHA,
                   help="LoRA alpha scaling (default: %(default)s).")
    p.add_argument("--lora_dropout", type=float, default=config.LORA_DROPOUT,
                   help="LoRA dropout (default: %(default)s).")
    p.add_argument("--lora_target_modules", type=str, nargs="+",
                   default=config.LORA_TARGET_MODULES,
                   help="LoRA target modules (default: %(default)s).")
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--batch_size", type=int, default=config.TRAIN_BATCH_SIZE)
    p.add_argument("--max_seq_len", type=int, default=config.MAX_SEQ_LEN)
    p.add_argument("--eval_steps", type=int, default=config.EVAL_STEPS)
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing (auto-enabled for long context).")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--data_dir", type=str, default=str(config.DATA_DIR))
    p.add_argument("--output_dir", type=str, default=str(config.CHECKPOINT_DIR))
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Override checkpoint directory (e.g. /mnt/d2/acp23ajh/dpmh/).")
    p.add_argument("--run_dir", type=str, default=None,
                   help="Timestamped run directory (e.g. results/2026-02-14_153000).")
    p.add_argument("--dry_run", action="store_true", help="Train for only 10 steps (pipeline testing).")
    p.add_argument(
        "--label_mode",
        choices=["binary", "multiclass"],
        default="binary",
        help="Label mode: 'binary' (default) or 'multiclass' (15-class BeaverTails).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint_dir:
        config.set_checkpoint_dir(args.checkpoint_dir)
    if args.run_dir:
        config.set_run_dir(args.run_dir)
        # Update data_dir to the run-scoped path if the user didn't
        # override it explicitly on the command line.
        if args.data_dir == str(config.PROJECT_ROOT / "data"):
            args.data_dir = str(config.DATA_DIR)

    print('All arguments:')
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    config.ensure_dirs()

    sizes = list(config.MODELS.keys()) if args.model_size == "all" else [args.model_size]
    splits = ["A", "B"] if args.split == "all" else [args.split]

    all_results = {}
    for size, split in product(sizes, splits):
        name = f"{size}_{split}"
        results = train_one_classifier(size, split, args)
        all_results[name] = results

    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE — Summary")
    print("=" * 70)
    for name, res in all_results.items():
        loss = res.get("eval_loss", "N/A")
        auc_val = res.get("eval_auc", "N/A")
        f1 = res.get("eval_f1", "N/A")
        print(f"  {name}: loss={loss}, AUC={auc_val}, F1={f1}")


if __name__ == "__main__":
    main()
