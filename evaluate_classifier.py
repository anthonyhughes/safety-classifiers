#!/usr/bin/env python3
"""
Evaluate a fine-tuned classifier on the held-out test set.

Computes accuracy, precision, recall, F1, AUC-ROC, and a confusion matrix.
For psychotherapy datasets, provides breakdowns by session_id (are later
sessions harder/easier to classify?) and by pairing characteristics.

Usage:
    python evaluate_classifier.py --run_dir results/2026-02-27_...
    python evaluate_classifier.py --run_dir results/2026-02-27_... --model_size 1b --split A
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

import config
from config import set_seed
from utils import (
    format_for_classification,
    format_multiturn_for_classification,
    load_tokenizer,
    print_table,
    save_json,
)


# =========================================================================== #
# Model loading (reuses the same logic as membership_inference.py)
# =========================================================================== #

def load_classifier(
    model_size: str, split: str,
    quantize: bool = False, num_labels: int = 2,
):
    """Load a fine-tuned classifier (LoRA or full, auto-detected)."""
    model_name = config.MODELS[model_size]
    ckpt = config.checkpoint_path(model_size, split)

    adapter_config_path = Path(str(ckpt)) / "adapter_config.json"
    use_lora = adapter_config_path.exists()

    load_kwargs: dict = {
        "num_labels": num_labels,
        "torch_dtype": torch.bfloat16,
    }
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

    tokenizer = load_tokenizer(model_name)

    if use_lora:
        from peft import PeftModel
        print(f"  Loading LoRA model from {ckpt}")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **load_kwargs,
        )
        base_model.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(base_model, str(ckpt))
    else:
        print(f"  Loading full model from {ckpt}")
        model = AutoModelForSequenceClassification.from_pretrained(
            str(ckpt), **load_kwargs,
        )
        model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    return model, tokenizer


# =========================================================================== #
# Scoring
# =========================================================================== #

@torch.no_grad()
def predict_dataset(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = config.INFERENCE_BATCH_SIZE,
    max_length: int = config.MAX_SEQ_LEN,
    task: str = "safety",
    conversations: list[str] | None = None,
    num_labels: int = 2,
    label_mode: str = "binary",
) -> dict[str, np.ndarray]:
    """Run the classifier and return predictions + probabilities."""

    # Format inputs
    if conversations is not None:
        formatted = []
        for conv_json in conversations:
            turns = json.loads(conv_json)
            formatted.append(format_multiturn_for_classification(turns, tokenizer))
    else:
        formatted = [
            format_for_classification(t, tokenizer, task=task, label_mode=label_mode)
            for t in texts
        ]

    # Auto-scale batch size for long-context datasets (matches membership_inference.py)
    if max_length > config.MAX_SEQ_LEN:
        scale = max(1, max_length // config.MAX_SEQ_LEN)
        batch_size = max(1, batch_size // scale)
        print(f"  Long-context: eval batch_size adjusted to {batch_size}")

    all_probs = []
    all_preds = []

    for i in tqdm(range(0, len(formatted), batch_size), desc="Scoring"):
        batch_texts = formatted[i : i + batch_size]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encodings = {k: v.to(model.device) for k, v in encodings.items()}

        outputs = model(**encodings)
        logits = outputs.logits.float()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()

        all_probs.append(probs)
        all_preds.append(preds)

    return {
        "predictions": np.concatenate(all_preds),
        "probabilities": np.concatenate(all_probs),
    }


# =========================================================================== #
# Evaluation
# =========================================================================== #

def evaluate(
    run_dir: str,
    model_size: str = "1b",
    split: str = "A",
    eval_split: str = "attack_eval",
    quantize: bool = False,
    batch_size: int = config.INFERENCE_BATCH_SIZE,
    max_seq_len: int | None = None,
) -> dict:
    """Evaluate a classifier on the held-out set and return a report."""

    config.set_run_dir(run_dir)

    # ----- Load data & metadata ----- #
    data_path = config.DATA_DIR / config.DATA_FILE
    if not data_path.exists():
        print(f"  ✗ Data not found at {data_path}")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    meta_path = config.DATA_DIR / "metadata.json"
    dataset_name = None
    label_mode = "binary"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            dataset_name = meta.get("dataset")
            if meta.get("label_mode") == "multiclass":
                label_mode = "multiclass"

    num_labels = config.get_num_labels(dataset_name, label_mode)
    label_names = config.get_label_names(dataset_name, label_mode=label_mode)
    if max_seq_len is not None:
        _max_seq_len = max_seq_len
    else:
        _max_seq_len = config.get_max_seq_len(dataset_name)
    multiturn = "conversations" in df.columns
    is_psycho = config.is_psychotherapy_mode(dataset_name)

    print(f"\n{'='*70}")
    print(f"  CLASSIFIER EVALUATION")
    print(f"{'='*70}")
    print(f"  Dataset      : {dataset_name}")
    print(f"  Label mode   : {label_mode}")
    print(f"  Classifier   : {model_size}_{split}")
    print(f"  Eval split   : {eval_split}")
    print(f"  Num labels   : {num_labels}")
    print(f"  Max seq len  : {_max_seq_len}")
    print(f"  Multi-turn   : {multiturn}")

    # ----- Filter to eval split ----- #
    eval_df = df[df["split"] == eval_split].reset_index(drop=True)
    if len(eval_df) == 0:
        print(f"  ✗ No examples in split '{eval_split}'. Available splits: "
              f"{df['split'].unique().tolist()}")
        sys.exit(1)

    print(f"  Eval size    : {len(eval_df)}")
    print(f"  Label dist   : {dict(eval_df['label'].value_counts().sort_index())}")

    # ----- Load model ----- #
    model, tokenizer = load_classifier(
        model_size, split, quantize=quantize, num_labels=num_labels,
    )

    # ----- Run predictions ----- #
    conversations = eval_df["conversations"].tolist() if multiturn else None
    results = predict_dataset(
        model, tokenizer,
        texts=eval_df["text"].tolist(),
        batch_size=batch_size,
        max_length=_max_seq_len,
        conversations=conversations,
        num_labels=num_labels,
        label_mode=label_mode,
    )

    predictions = results["predictions"]
    probabilities = results["probabilities"]
    labels = eval_df["label"].values

    # ----- Overall metrics ----- #
    report: dict = {
        "dataset": dataset_name,
        "classifier": f"{model_size}_{split}",
        "eval_split": eval_split,
        "n_examples": len(eval_df),
    }

    if num_labels <= 2:
        report["accuracy"] = float(accuracy_score(labels, predictions))
        report["precision"] = float(precision_score(labels, predictions, zero_division=0))
        report["recall"] = float(recall_score(labels, predictions, zero_division=0))
        report["f1"] = float(f1_score(labels, predictions, zero_division=0))
        if len(np.unique(labels)) > 1:
            report["auc_roc"] = float(roc_auc_score(labels, probabilities[:, 1]))
    else:
        report["accuracy"] = float(accuracy_score(labels, predictions))
        report["precision_macro"] = float(precision_score(
            labels, predictions, average="macro", zero_division=0))
        report["recall_macro"] = float(recall_score(
            labels, predictions, average="macro", zero_division=0))
        report["f1_macro"] = float(f1_score(
            labels, predictions, average="macro", zero_division=0))
        report["f1_weighted"] = float(f1_score(
            labels, predictions, average="weighted", zero_division=0))
        if len(np.unique(labels)) > 1:
            try:
                report["auc_roc"] = float(roc_auc_score(
                    labels, probabilities, multi_class="ovr", average="macro"))
            except ValueError:
                pass

    report["confusion_matrix"] = confusion_matrix(labels, predictions).tolist()

    # Sklearn classification report as a dict
    clf_report = classification_report(
        labels, predictions, target_names=[label_names.get(i, str(i)) for i in range(num_labels)],
        output_dict=True, zero_division=0,
    )
    report["classification_report"] = clf_report

    # ----- Print results ----- #
    print(f"\n{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}")
    print(f"  Accuracy  : {report['accuracy']:.4f}")
    if num_labels <= 2:
        print(f"  Precision : {report['precision']:.4f}")
        print(f"  Recall    : {report['recall']:.4f}")
        print(f"  F1        : {report['f1']:.4f}")
    else:
        print(f"  Prec (m)  : {report['precision_macro']:.4f}")
        print(f"  Recall (m): {report['recall_macro']:.4f}")
        print(f"  F1 (macro): {report['f1_macro']:.4f}")
        print(f"  F1 (wt)   : {report['f1_weighted']:.4f}")
    if "auc_roc" in report:
        print(f"  AUC-ROC   : {report['auc_roc']:.4f}")

    cm = report["confusion_matrix"]
    print(f"\n  Confusion Matrix:")
    print(f"    {'':>20s}  Predicted")
    for i, row in enumerate(cm):
        lbl = label_names.get(i, str(i))
        print(f"    True {lbl:>12s}: {row}")

    print(f"\n  Classification Report:")
    print(classification_report(
        labels, predictions,
        target_names=[label_names.get(i, str(i)) for i in range(num_labels)],
        zero_division=0,
    ))

    # ----- Psychotherapy-specific breakdowns ----- #
    if is_psycho and "session_id" in eval_df.columns:
        print(f"\n{'='*70}")
        print("  BREAKDOWN BY SESSION ID")
        print(f"{'='*70}")

        eval_df = eval_df.copy()
        eval_df["predicted"] = predictions
        eval_df["correct"] = (predictions == labels).astype(int)
        if probabilities.shape[1] == 2:
            eval_df["prob_positive"] = probabilities[:, 1]

        session_metrics = []
        for sid in sorted(eval_df["session_id"].unique()):
            mask = eval_df["session_id"] == sid
            sub = eval_df[mask]
            sub_labels = sub["label"].values
            sub_preds = sub["predicted"].values
            n = len(sub)
            n_pos = int(sub_labels.sum())
            acc = float(accuracy_score(sub_labels, sub_preds))
            f1 = float(f1_score(sub_labels, sub_preds, zero_division=0))
            auc = None
            if probabilities.shape[1] == 2 and len(np.unique(sub_labels)) > 1:
                auc = float(roc_auc_score(sub_labels, sub.loc[mask, "prob_positive"].values))
            session_metrics.append({
                "session_id": int(sid),
                "n": n,
                "n_positive": n_pos,
                "accuracy": acc,
                "f1": f1,
                "auc_roc": auc,
            })

        report["by_session_id"] = session_metrics

        headers = ["Session", "N", "Pos", "Acc", "F1", "AUC"]
        rows = [
            [str(m["session_id"]), str(m["n"]), str(m["n_positive"]),
             f"{m['accuracy']:.3f}", f"{m['f1']:.3f}",
             f"{m['auc_roc']:.3f}" if m["auc_roc"] is not None else "N/A"]
            for m in session_metrics
        ]
        print_table(headers, rows, title="Metrics by Session")

    if is_psycho and "num_sessions_context" in eval_df.columns:
        print(f"\n{'='*70}")
        print("  BREAKDOWN BY CONTEXT DEPTH (sliding-window)")
        print(f"{'='*70}")

        eval_df_copy = eval_df.copy() if "predicted" not in eval_df.columns else eval_df
        if "predicted" not in eval_df_copy.columns:
            eval_df_copy["predicted"] = predictions
            eval_df_copy["correct"] = (predictions == labels).astype(int)

        context_metrics = []
        for depth in sorted(eval_df_copy["num_sessions_context"].unique()):
            mask = eval_df_copy["num_sessions_context"] == depth
            sub = eval_df_copy[mask]
            sub_labels = sub["label"].values
            sub_preds = sub["predicted"].values
            n = len(sub)
            acc = float(accuracy_score(sub_labels, sub_preds))
            f1 = float(f1_score(sub_labels, sub_preds, zero_division=0))
            context_metrics.append({
                "context_depth": int(depth),
                "n": n,
                "accuracy": acc,
                "f1": f1,
            })

        report["by_context_depth"] = context_metrics

        headers = ["Depth", "N", "Acc", "F1"]
        rows = [
            [str(m["context_depth"]), str(m["n"]),
             f"{m['accuracy']:.3f}", f"{m['f1']:.3f}"]
            for m in context_metrics
        ]
        print_table(headers, rows, title="Metrics by Context Depth")

    # ----- Save results ----- #
    classifier_name = f"{model_size}_{split}"

    # Save to scores dir (picked up by analyze.py)
    test_results_path = config.SCORES_DIR / f"test_results_{classifier_name}.json"
    save_json(report, test_results_path)
    print(f"\n  Test results saved to {test_results_path}")

    # Also save to evaluation dir for backwards compatibility
    eval_dir = Path(run_dir) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_path = eval_dir / f"classification_report_{classifier_name}.json"
    save_json(report, report_path)
    print(f"  Full report saved to {report_path}")

    return report


# =========================================================================== #
# CLI
# =========================================================================== #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a fine-tuned classifier on the held-out test set.",
    )
    p.add_argument("--run_dir", type=str, required=True,
                   help="Timestamped run directory (e.g. results/2026-02-27_...).")
    p.add_argument("--model_size", type=str, default="1b",
                   choices=list(config.MODELS.keys()),
                   help="Model size to evaluate (default: 1b).")
    p.add_argument("--split", type=str, default="A",
                   choices=["A", "B", "all"],
                   help="Training split to evaluate (default: A). Use 'all' for A+B.")
    p.add_argument("--eval_split", type=str, default="attack_eval",
                   help="Data split to evaluate on (default: attack_eval).")
    p.add_argument("--max_seq_len", type=int, default=None,
                   help="Max sequence length (auto-detected from dataset if not set).")
    p.add_argument("--quantize", action="store_true",
                   help="Load model with 4-bit quantization.")
    p.add_argument("--batch_size", type=int, default=config.INFERENCE_BATCH_SIZE,
                   help="Inference batch size (default: %(default)s).")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Override checkpoint directory (e.g. /mnt/d2/acp23ajh/dpmh/).")
    p.add_argument("--dry_run", action="store_true",
                   help="Quick smoke test (no-op for evaluation, accepted for pipeline compat).")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    if args.checkpoint_dir:
        config.set_checkpoint_dir(args.checkpoint_dir)

    splits = ["A", "B"] if args.split == "all" else [args.split]

    for split in splits:
        report = evaluate(
            run_dir=args.run_dir,
            model_size=args.model_size,
            split=split,
            eval_split=args.eval_split,
            quantize=args.quantize,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
        )
