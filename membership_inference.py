#!/usr/bin/env python3
"""
Phase 3 — Membership inference attacks.

For every classifier, score all data (A_train, B_train, attack_cal,
attack_eval) and then run four membership inference attacks:

  1. Loss-based (Yeom et al. 2018)
  2. Confidence-based
  3. Reference-model / LiRA-style (Carlini et al. 2022)
  4. Shadow-model (logistic regression on calibration data)

Usage:
    python membership_inference.py --classifier all
    python membership_inference.py --classifier 1b_A
    python membership_inference.py --classifier all --dry_run
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

import config
from config import set_seed
from utils import (
    bootstrap_ci,
    compute_mi_metrics,
    format_for_classification,
    format_multiturn_for_classification,
    load_tokenizer,
    print_table,
    save_json,
)


# =========================================================================== #
# Score generation
# =========================================================================== #

def load_classifier(model_size: str, split: str, quantize: bool = False,
                    num_labels: int = 2):
    """Load a fine-tuned classifier (LoRA or full model, auto-detected)."""
    model_name = config.MODELS[model_size]
    ckpt = config.checkpoint_path(model_size, split)

    # Auto-detect training mode: LoRA if adapter_config.json exists
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
            model_name, **load_kwargs
        )
        base_model.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(base_model, str(ckpt))
    else:
        # Load full fine-tuned model directly from checkpoint
        print(f"  Loading full (non-LoRA) model from {ckpt}")
        model = AutoModelForSequenceClassification.from_pretrained(
            str(ckpt), **load_kwargs
        )
        model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    return model, tokenizer


@torch.no_grad()
def score_dataset(
    model,
    tokenizer,
    texts: list[str],
    labels: list[int],
    batch_size: int = config.INFERENCE_BATCH_SIZE,
    max_length: int = config.MAX_SEQ_LEN,
    task: str = "safety",
    conversations: list[str] | None = None,
    num_labels: int = 2,
) -> dict[str, np.ndarray]:
    """
    Run classifier on a list of texts and return per-example scores.

    When *conversations* is provided (list of JSON-encoded turn lists),
    multi-turn chat-template formatting is used instead of single-turn.

    Returns dict with arrays: predicted_prob, loss, predicted_label, correct,
    and full_probs (N, num_labels) for logit-vector attacks.
    """
    all_probs = []          # scalar: P(true class) for binary; P(positive) for binary
    all_full_probs = []     # full softmax vectors (N, num_labels)
    all_losses = []
    all_preds = []
    all_correct = []

    multiturn = conversations is not None
    # Determine label_mode from num_labels
    _label_mode = "multiclass" if num_labels > 2 else "binary"

    n = len(texts)
    for start in tqdm(range(0, n, batch_size), desc="  Scoring"):
        end = min(start + batch_size, n)

        if multiturn:
            import json as _json
            batch_formatted = []
            for conv_json in conversations[start:end]:
                turns = _json.loads(conv_json)
                batch_formatted.append(
                    format_multiturn_for_classification(turns, tokenizer, task=task)
                )
        else:
            batch_formatted = [
                format_for_classification(
                    t, tokenizer, task=task, label_mode=_label_mode,
                )
                for t in texts[start:end]
            ]
        batch_labels = labels[start:end]

        encodings = tokenizer(
            batch_formatted,
            truncation=True,
            padding="longest",
            max_length=max_length,
            return_tensors="pt",
        )

        # Move to model device
        device = next(model.parameters()).device
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        label_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits.float()  # (batch, 2)

        # Softmax probabilities
        probs = F.softmax(logits, dim=-1)  # (batch, num_labels)

        if num_labels <= 2:
            prob_positive = probs[:, 1].cpu().numpy()  # P(harmful)
        else:
            # Multi-class: P(true class) for each example
            prob_positive = probs[
                torch.arange(probs.size(0)), label_tensor
            ].cpu().numpy()

        # Save full probability vectors for logit-vector attack
        full_probs_batch = probs.cpu().numpy()  # (batch, num_labels)

        # Per-example cross-entropy loss
        losses = F.cross_entropy(logits, label_tensor, reduction="none").cpu().numpy()

        # Predictions
        preds = logits.argmax(dim=-1).cpu().numpy()
        correct = (preds == np.array(batch_labels)).astype(int)

        all_probs.extend(prob_positive.tolist())
        all_full_probs.append(full_probs_batch)
        all_losses.extend(losses.tolist())
        all_preds.extend(preds.tolist())
        all_correct.extend(correct.tolist())

    return {
        "predicted_prob": np.array(all_probs),
        "loss": np.array(all_losses),
        "predicted_label": np.array(all_preds),
        "correct": np.array(all_correct),
        "full_probs": np.vstack(all_full_probs),  # (N, num_labels)
    }


def generate_scores(
    classifier_name: str,
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Generate scores for one classifier on all relevant splits."""
    model_size, split = config.parse_classifier_name(classifier_name)
    print(f"\n--- Generating scores for {classifier_name} ---")

    # Auto-detect task
    task = getattr(args, "task", "auto")
    if task == "auto":
        task = "language" if "language" in df.columns else "safety"

    # Detect multi-turn mode
    multiturn = "conversations" in df.columns

    # Determine num_labels
    num_labels = getattr(args, "_num_labels", 2)

    # Override task for pooled dataset (generic system prompt)
    dataset_name = getattr(args, "_dataset_name", None)
    if dataset_name == "pooled":
        task = "pooled"

    model, tokenizer = load_classifier(
        model_size, split, quantize=args.quantize, num_labels=num_labels,
    )

    # Select splits to score
    splits_to_score = ["A_train", "B_train", "attack_cal", "attack_eval"]
    sub = df[df["split"].isin(splits_to_score)].reset_index(drop=True)

    if args.dry_run:
        sub = sub.head(100)

    texts = sub["text"].tolist()
    labels = sub["label"].tolist()
    conversations = sub["conversations"].tolist() if multiturn else None

    # Determine effective max_length and batch_size
    max_length = getattr(args, "max_seq_len", config.MAX_SEQ_LEN)
    effective_batch = args.batch_size
    if max_length > config.MAX_SEQ_LEN:
        scale = max(1, max_length // config.MAX_SEQ_LEN)
        effective_batch = max(1, args.batch_size // scale)
        print(f"  Long-context: batch_size adjusted to {effective_batch}")

    scores = score_dataset(
        model, tokenizer, texts, labels,
        batch_size=effective_batch,
        max_length=max_length,
        task=task,
        conversations=conversations,
        num_labels=num_labels,
    )

    result_df = pd.DataFrame({
        "conversation_index": sub["original_index"].values,
        "true_label": sub["label"].values,
        "split": sub["split"].values,
        "classifier": classifier_name,
        "predicted_prob": scores["predicted_prob"],
        "loss": scores["loss"],
        "predicted_label": scores["predicted_label"],
        "correct": scores["correct"],
    })

    # Store full probability vectors for logit-vector attack
    full_probs = scores["full_probs"]  # (N, num_labels)
    full_probs_path = config.SCORES_DIR / f"full_probs_{classifier_name}.npy"
    np.save(full_probs_path, full_probs)
    print(f"  Saved full probability vectors ({full_probs.shape}) to {full_probs_path}")

    # Preserve language column if present (for breakdown_by_language)
    if "language" in sub.columns:
        result_df["language"] = sub["language"].values

    # Preserve BeaverTails category column if present (for breakdown_by_bt_category)
    if "bt_category" in sub.columns:
        result_df["bt_category"] = sub["bt_category"].values

    # Preserve source column if present (for breakdown_by_source)
    if "source" in sub.columns:
        result_df["source"] = sub["source"].values

    # Preserve token_count column if present (for breakdown_by_token_length)
    if "token_count" in sub.columns:
        result_df["token_count"] = sub["token_count"].values

    # Preserve num_turns column if present (for breakdown_by_token_length)
    if "num_turns" in sub.columns:
        result_df["num_turns"] = sub["num_turns"].values

    # Preserve emotion_intensity column (for breakdown_by_emotion_intensity)
    if "emotion_intensity" in sub.columns:
        result_df["emotion_intensity"] = sub["emotion_intensity"].values

    # Preserve problem_type column (for breakdown_by_problem_type)
    if "problem_type" in sub.columns:
        result_df["problem_type"] = sub["problem_type"].values

    # Save individual classifier scores
    out_path = config.scores_path(classifier_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(out_path, index=False)
    print(f"  Saved {len(result_df)} scores to {out_path}")

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result_df


# =========================================================================== #
# Attack implementations
# =========================================================================== #

def _true_class_prob(predicted_prob: np.ndarray, true_label: np.ndarray) -> np.ndarray:
    """P(true class).

    For binary (predicted_prob is a scalar P(class=1)):
        P(true class) = predicted_prob if label=1, else 1 - predicted_prob.
    For multi-class (predicted_prob is already P(true class)):
        returned as-is.

    The caller is responsible for passing the correct form of predicted_prob.
    In binary mode, pass ``probs[:, 1]`` (probability of class 1).
    In multiclass mode, pass the true-class probability directly.
    """
    # If labels are all 0-or-1 and probs look like binary P(class=1),
    # apply the binary formula.  Otherwise assume probs are already
    # true-class probabilities.
    unique_labels = np.unique(true_label)
    if set(unique_labels).issubset({0, 1}):
        return np.where(true_label == 1, predicted_prob, 1.0 - predicted_prob)
    # Multi-class: predicted_prob should already be P(true class)
    return predicted_prob


def attack_loss_based(
    scores_df: pd.DataFrame,
    classifier_name: str,
) -> dict:
    """
    Attack 1 — Loss-based (Yeom et al. 2018).

    Members tend to have lower loss. Signal = -loss (higher → more likely member).
    """
    _, split = config.parse_classifier_name(classifier_name)
    other_split = "B" if split == "A" else "A"

    clf_scores = scores_df[scores_df["classifier"] == classifier_name]
    members = clf_scores[clf_scores["split"] == f"{split}_train"]
    nonmembers = clf_scores[clf_scores["split"] == f"{other_split}_train"]

    member_signal = -members["loss"].values  # negative loss: higher = more likely member
    nonmember_signal = -nonmembers["loss"].values

    metrics = compute_mi_metrics(member_signal, nonmember_signal)
    cis = bootstrap_ci(member_signal, nonmember_signal)

    return {
        "attack": "loss_based",
        "classifier": classifier_name,
        "n_members": len(members),
        "n_nonmembers": len(nonmembers),
        **metrics,
        "confidence_intervals": cis,
    }


def attack_confidence_based(
    scores_df: pd.DataFrame,
    classifier_name: str,
) -> dict:
    """
    Attack 2 — Confidence-based.

    Signal = P(true class). Members get more confident correct predictions.
    """
    _, split = config.parse_classifier_name(classifier_name)
    other_split = "B" if split == "A" else "A"

    clf_scores = scores_df[scores_df["classifier"] == classifier_name]
    members = clf_scores[clf_scores["split"] == f"{split}_train"]
    nonmembers = clf_scores[clf_scores["split"] == f"{other_split}_train"]

    member_signal = _true_class_prob(members["predicted_prob"].values, members["true_label"].values)
    nonmember_signal = _true_class_prob(nonmembers["predicted_prob"].values, nonmembers["true_label"].values)

    metrics = compute_mi_metrics(member_signal, nonmember_signal)
    cis = bootstrap_ci(member_signal, nonmember_signal)

    return {
        "attack": "confidence_based",
        "classifier": classifier_name,
        "n_members": len(members),
        "n_nonmembers": len(nonmembers),
        **metrics,
        "confidence_intervals": cis,
    }


def attack_reference_model(
    scores_df: pd.DataFrame,
    classifier_name: str,
) -> dict:
    """
    Attack 3 — Reference-model / LiRA-style (Carlini et al. 2022).

    For classifier X_A, use classifier X_B as reference (same size, different split).
    Signal = log(P_target(true class) / P_reference(true class)).
    """
    model_size, split = config.parse_classifier_name(classifier_name)
    other_split = "B" if split == "A" else "A"
    ref_classifier = f"{model_size}_{other_split}"

    # Target classifier scores
    target_scores = scores_df[scores_df["classifier"] == classifier_name].copy()
    target_scores = target_scores.set_index(["conversation_index", "split"])

    # Reference classifier scores
    ref_scores = scores_df[scores_df["classifier"] == ref_classifier].copy()
    ref_scores = ref_scores.set_index(["conversation_index", "split"])

    # Align on conversation_index and split
    common_idx = target_scores.index.intersection(ref_scores.index)
    target_aligned = target_scores.loc[common_idx]
    ref_aligned = ref_scores.loc[common_idx]

    # Compute P(true class) for both
    eps = 1e-7
    target_p_true = np.clip(
        _true_class_prob(target_aligned["predicted_prob"].values, target_aligned["true_label"].values),
        eps, 1 - eps,
    )
    ref_p_true = np.clip(
        _true_class_prob(ref_aligned["predicted_prob"].values, ref_aligned["true_label"].values),
        eps, 1 - eps,
    )

    log_ratio = np.log(target_p_true / ref_p_true)

    # Split into members / nonmembers of the target classifier
    splits = target_aligned.index.get_level_values("split")
    member_mask = splits == f"{split}_train"
    nonmember_mask = splits == f"{other_split}_train"

    member_signal = log_ratio[member_mask]
    nonmember_signal = log_ratio[nonmember_mask]

    metrics = compute_mi_metrics(member_signal, nonmember_signal)
    cis = bootstrap_ci(member_signal, nonmember_signal)

    return {
        "attack": "reference_model",
        "classifier": classifier_name,
        "reference_classifier": ref_classifier,
        "n_members": int(member_mask.sum()),
        "n_nonmembers": int(nonmember_mask.sum()),
        **metrics,
        "confidence_intervals": cis,
    }


def attack_shadow_model(
    scores_df: pd.DataFrame,
    classifier_name: str,
) -> dict:
    """
    Attack 4 — Shadow model.

    Train a logistic regression on attack_cal using the target classifier's
    scores as features, with membership in target training set as the label.
    Evaluate on attack_eval + training sets.
    """
    _, split = config.parse_classifier_name(classifier_name)
    other_split = "B" if split == "A" else "A"

    clf_scores = scores_df[scores_df["classifier"] == classifier_name].copy()

    # ----- Build calibration data -----
    # On calibration set, we know nothing about "membership" directly, so
    # we use the training splits for calibration: a portion as shadow train, rest as eval.
    # But the cleaner design: train on attack_cal features where we define
    # pseudo-membership from the target's behavior patterns.
    # Actually, the standard shadow approach: train a binary classifier to predict
    # membership using (predicted_prob, loss, true_label) as features.
    # For calibration: we use examples from A_train (members) and B_train (non-members)
    # from attack_cal's perspective... but attack_cal isn't in either training set.
    #
    # Better approach: use a subset of A_train + B_train for training the attack model,
    # and evaluate on the rest. We'll use attack_cal to augment non-member examples.

    # Training set members
    members_all = clf_scores[clf_scores["split"] == f"{split}_train"].copy()
    nonmembers_all = clf_scores[clf_scores["split"] == f"{other_split}_train"].copy()

    # Use half of each for shadow training, half for evaluation
    rng = np.random.RandomState(config.SEED)
    n_cal_mem = len(members_all) // 2
    n_cal_non = len(nonmembers_all) // 2

    mem_idx = rng.permutation(len(members_all))
    non_idx = rng.permutation(len(nonmembers_all))

    cal_mem = members_all.iloc[mem_idx[:n_cal_mem]]
    eval_mem = members_all.iloc[mem_idx[n_cal_mem:]]
    cal_non = nonmembers_all.iloc[non_idx[:n_cal_non]]
    eval_non = nonmembers_all.iloc[non_idx[n_cal_non:]]

    # Also add attack_cal examples as non-members (they're not in any training set)
    attack_cal = clf_scores[clf_scores["split"] == "attack_cal"]
    cal_non = pd.concat([cal_non, attack_cal], ignore_index=True)

    def _make_features(sub_df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            sub_df["predicted_prob"].values,
            sub_df["loss"].values,
            sub_df["true_label"].values,
        ])

    # Train logistic regression
    X_train = np.vstack([_make_features(cal_mem), _make_features(cal_non)])
    y_train = np.concatenate([np.ones(len(cal_mem)), np.zeros(len(cal_non))])

    lr = LogisticRegression(max_iter=1000, random_state=config.SEED)
    lr.fit(X_train, y_train)

    # Evaluate: predict membership probability on held-out members and non-members
    X_eval_mem = _make_features(eval_mem)
    X_eval_non = _make_features(eval_non)

    member_signal = lr.predict_proba(X_eval_mem)[:, 1]
    nonmember_signal = lr.predict_proba(X_eval_non)[:, 1]

    metrics = compute_mi_metrics(member_signal, nonmember_signal)
    cis = bootstrap_ci(member_signal, nonmember_signal)

    return {
        "attack": "shadow_model",
        "classifier": classifier_name,
        "n_members": len(eval_mem),
        "n_nonmembers": len(eval_non),
        "cal_members": len(cal_mem),
        "cal_nonmembers": len(cal_non),
        **metrics,
        "confidence_intervals": cis,
    }


# =========================================================================== #
# Breakdown analyses
# =========================================================================== #

def breakdown_by_label(
    scores_df: pd.DataFrame,
    classifier_name: str,
    attack_fn,
    attack_name: str,
    task: str = "safety",
    dataset: str | None = None,
    label_mode: str = "binary",
) -> dict:
    """Compute MI-AUC separately for each label value.

    Label names adapt to the task / dataset:
      - safety: harmful (1) / benign (0)
      - language: english (1) / non_english (0)
      - beavertails: unsafe (1) / safe (0)  (binary)
      - beavertails multiclass: safe (0), 14 harm categories (1-14)
    """
    label_names = config.get_label_names(dataset, label_mode=label_mode)
    if dataset is None:
        # Fallback to task-based selection for backwards compat
        label_names = (config.LANGUAGE_LABEL_NAMES if task == "language"
                       else config.SAFETY_LABEL_NAMES)
    _, split = config.parse_classifier_name(classifier_name)
    other_split = "B" if split == "A" else "A"

    clf_scores = scores_df[scores_df["classifier"] == classifier_name]

    results = {}
    for label_val, label_name in label_names.items():
        sub = clf_scores[clf_scores["true_label"] == label_val]
        members = sub[sub["split"] == f"{split}_train"]
        nonmembers = sub[sub["split"] == f"{other_split}_train"]

        if len(members) < 10 or len(nonmembers) < 10:
            results[label_name] = {"mi_auc": float("nan"), "n_mem": len(members), "n_non": len(nonmembers)}
            continue

        if attack_name == "loss_based":
            mem_signal = -members["loss"].values
            non_signal = -nonmembers["loss"].values
        elif attack_name == "confidence_based":
            mem_signal = _true_class_prob(members["predicted_prob"].values, members["true_label"].values)
            non_signal = _true_class_prob(nonmembers["predicted_prob"].values, nonmembers["true_label"].values)
        else:
            continue

        metrics = compute_mi_metrics(mem_signal, non_signal)
        results[label_name] = {"n_mem": len(members), "n_non": len(nonmembers), **metrics}

    return results


def breakdown_by_confidence(
    scores_df: pd.DataFrame,
    classifier_name: str,
    n_bins: int = 10,
) -> list[dict]:
    """Compute MI-AUC per confidence decile."""
    _, split = config.parse_classifier_name(classifier_name)
    other_split = "B" if split == "A" else "A"

    clf_scores = scores_df[scores_df["classifier"] == classifier_name].copy()
    clf_scores["confidence"] = np.abs(clf_scores["predicted_prob"].values - 0.5)

    # Restrict to training splits
    train_data = clf_scores[clf_scores["split"].isin([f"{split}_train", f"{other_split}_train"])].copy()
    train_data["is_member"] = (train_data["split"] == f"{split}_train").astype(int)

    # Bin by confidence
    train_data["conf_bin"] = pd.qcut(train_data["confidence"], q=n_bins, labels=False, duplicates="drop")

    bins_results = []
    for bin_id in sorted(train_data["conf_bin"].unique()):
        bin_data = train_data[train_data["conf_bin"] == bin_id]
        members = bin_data[bin_data["is_member"] == 1]
        nonmembers = bin_data[bin_data["is_member"] == 0]

        if len(members) < 5 or len(nonmembers) < 5:
            continue

        # Use loss-based signal for this breakdown
        mem_signal = -members["loss"].values
        non_signal = -nonmembers["loss"].values
        metrics = compute_mi_metrics(mem_signal, non_signal)

        bins_results.append({
            "bin": int(bin_id),
            "confidence_mean": float(bin_data["confidence"].mean()),
            "n_members": len(members),
            "n_nonmembers": len(nonmembers),
            **metrics,
        })

    return bins_results


def breakdown_by_language(
    scores_df: pd.DataFrame,
    classifier_name: str,
    min_per_group: int = 20,
) -> list[dict]:
    """Compute MI-AUC per individual language.

    Only meaningful when the parquet contains a ``language`` column
    (i.e. WildChat language-mode data).
    """
    if "language" not in scores_df.columns:
        return []

    _, split = config.parse_classifier_name(classifier_name)
    other_split = "B" if split == "A" else "A"

    clf_scores = scores_df[scores_df["classifier"] == classifier_name].copy()
    train_data = clf_scores[clf_scores["split"].isin(
        [f"{split}_train", f"{other_split}_train"]
    )].copy()
    train_data["is_member"] = (train_data["split"] == f"{split}_train").astype(int)

    results = []
    for lang, group in train_data.groupby("language"):
        members = group[group["is_member"] == 1]
        nonmembers = group[group["is_member"] == 0]

        if len(members) < min_per_group or len(nonmembers) < min_per_group:
            continue

        # Loss-based signal
        mem_signal = -members["loss"].values
        non_signal = -nonmembers["loss"].values
        metrics = compute_mi_metrics(mem_signal, non_signal)

        results.append({
            "language": str(lang),
            "n_members": len(members),
            "n_nonmembers": len(nonmembers),
            **metrics,
        })

    # Sort by MI-AUC descending
    results.sort(key=lambda r: r.get("mi_auc", 0.5), reverse=True)
    return results


def breakdown_by_bt_category(
    scores_df: pd.DataFrame,
    classifier_name: str,
    min_per_group: int = 20,
) -> list[dict]:
    """Compute MI-AUC per BeaverTails harm category.

    Only meaningful when the parquet contains a ``bt_category`` column
    (i.e. BeaverTails data).  Each sample has exactly one primary category
    (first True in config order) or ``safe``.
    """
    if "bt_category" not in scores_df.columns:
        return []

    _, split = config.parse_classifier_name(classifier_name)
    other_split = "B" if split == "A" else "A"

    clf_scores = scores_df[scores_df["classifier"] == classifier_name].copy()
    train_data = clf_scores[clf_scores["split"].isin(
        [f"{split}_train", f"{other_split}_train"]
    )].copy()
    train_data["is_member"] = (train_data["split"] == f"{split}_train").astype(int)

    results = []
    for cat, group in train_data.groupby("bt_category"):
        members = group[group["is_member"] == 1]
        nonmembers = group[group["is_member"] == 0]

        if len(members) < min_per_group or len(nonmembers) < min_per_group:
            continue

        # Loss-based signal
        mem_signal = -members["loss"].values
        non_signal = -nonmembers["loss"].values
        metrics = compute_mi_metrics(mem_signal, non_signal)

        results.append({
            "category": str(cat),
            "n_members": len(members),
            "n_nonmembers": len(nonmembers),
            **metrics,
        })

    # Sort by MI-AUC descending
    results.sort(key=lambda r: r.get("mi_auc", 0.5), reverse=True)
    return results


def attack_per_category(
    scores_df: pd.DataFrame,
    classifier_name: str,
    min_per_group: int = 20,
) -> dict:
    """Per-category MIA: for each harm category, run loss-based and
    confidence-based attacks on the subset of examples with that label.

    Uses ``bt_category`` column (available in both binary and multiclass
    modes) or ``true_label`` for multiclass.

    Returns a dict keyed by category name, each with loss-based and
    confidence-based AUC, TPR@FPR metrics, and sample counts.
    """
    _, split = config.parse_classifier_name(classifier_name)
    other_split = "B" if split == "A" else "A"

    clf_scores = scores_df[scores_df["classifier"] == classifier_name].copy()
    train_data = clf_scores[clf_scores["split"].isin(
        [f"{split}_train", f"{other_split}_train"]
    )].copy()
    train_data["is_member"] = (train_data["split"] == f"{split}_train").astype(int)

    # Determine grouping column
    if "bt_category" in train_data.columns:
        group_col = "bt_category"
    else:
        group_col = "true_label"

    results = {}
    for cat, group in train_data.groupby(group_col):
        members = group[group["is_member"] == 1]
        nonmembers = group[group["is_member"] == 0]

        if len(members) < min_per_group or len(nonmembers) < min_per_group:
            results[str(cat)] = {
                "n_members": len(members),
                "n_nonmembers": len(nonmembers),
                "skipped": True,
            }
            continue

        cat_result: dict = {
            "n_members": len(members),
            "n_nonmembers": len(nonmembers),
        }

        # Loss-based
        mem_loss = -members["loss"].values
        non_loss = -nonmembers["loss"].values
        loss_metrics = compute_mi_metrics(mem_loss, non_loss)
        cat_result["loss_based"] = loss_metrics

        # Confidence-based
        mem_conf = _true_class_prob(
            members["predicted_prob"].values, members["true_label"].values,
        )
        non_conf = _true_class_prob(
            nonmembers["predicted_prob"].values, nonmembers["true_label"].values,
        )
        conf_metrics = compute_mi_metrics(mem_conf, non_conf)
        cat_result["confidence_based"] = conf_metrics

        results[str(cat)] = cat_result

    return results


def breakdown_by_source(
    scores_df: pd.DataFrame,
    classifier_name: str,
    min_per_group: int = 20,
) -> list[dict]:
    """Compute MI-AUC per data source (e.g. xguard vs wildchat).

    Only meaningful when the parquet contains a ``source`` column
    (i.e. xguard-multiturn data).
    """
    if "source" not in scores_df.columns:
        return []

    _, split = config.parse_classifier_name(classifier_name)
    other_split = "B" if split == "A" else "A"

    clf_scores = scores_df[scores_df["classifier"] == classifier_name].copy()
    train_data = clf_scores[clf_scores["split"].isin(
        [f"{split}_train", f"{other_split}_train"]
    )].copy()
    train_data["is_member"] = (train_data["split"] == f"{split}_train").astype(int)

    results = []
    for src, group in train_data.groupby("source"):
        members = group[group["is_member"] == 1]
        nonmembers = group[group["is_member"] == 0]

        if len(members) < min_per_group or len(nonmembers) < min_per_group:
            continue

        # Loss-based signal
        mem_signal = -members["loss"].values
        non_signal = -nonmembers["loss"].values
        metrics = compute_mi_metrics(mem_signal, non_signal)

        results.append({
            "source": str(src),
            "n_members": len(members),
            "n_nonmembers": len(nonmembers),
            **metrics,
        })

    results.sort(key=lambda r: r.get("mi_auc", 0.5), reverse=True)
    return results


def breakdown_by_token_length(
    scores_df: pd.DataFrame,
    classifier_name: str,
    n_bins: int = 4,
    min_per_group: int = 20,
) -> list[dict]:
    """Compute MI-AUC per token-length bin (loss-based + reference-model).

    Bins conversations by ``token_count`` using quantile binning so that
    each bin has roughly equal sample counts.  Only meaningful when the
    scores parquet contains a ``token_count`` column (i.e. xguard-multiturn
    data or any dataset that records token counts in the data parquet).

    When both A and B classifier scores are available, also computes a
    reference-model signal per bin (log-likelihood ratio).

    Returns a list of dicts, one per bin, with keys:
      bin, token_count_min, token_count_max, token_count_mean,
      n_members, n_nonmembers, mi_auc, tpr_at_fpr_1pct, tpr_at_fpr_5pct,
      and optionally ref_mi_auc, ref_tpr_at_fpr_1pct, ref_tpr_at_fpr_5pct
    """
    if "token_count" not in scores_df.columns:
        return []

    model_size, split = config.parse_classifier_name(classifier_name)
    other_split = "B" if split == "A" else "A"
    ref_classifier = f"{model_size}_{other_split}"
    has_ref = ref_classifier in scores_df["classifier"].unique()

    clf_scores = scores_df[scores_df["classifier"] == classifier_name].copy()
    train_data = clf_scores[clf_scores["split"].isin(
        [f"{split}_train", f"{other_split}_train"]
    )].copy()
    train_data["is_member"] = (train_data["split"] == f"{split}_train").astype(int)

    # If reference classifier available, build aligned log-ratio signal
    if has_ref:
        ref_scores = scores_df[scores_df["classifier"] == ref_classifier].copy()
        ref_train = ref_scores[ref_scores["split"].isin(
            [f"{split}_train", f"{other_split}_train"]
        )]
        # Merge on conversation_index + split to align target & reference
        ref_cols = ref_train[["conversation_index", "split", "predicted_prob", "true_label"]].copy()
        ref_cols = ref_cols.rename(columns={
            "predicted_prob": "ref_predicted_prob",
            "true_label": "ref_true_label",
        })
        train_data = train_data.merge(ref_cols, on=["conversation_index", "split"], how="inner")

        eps = 1e-7
        target_p_true = np.clip(
            _true_class_prob(train_data["predicted_prob"].values, train_data["true_label"].values),
            eps, 1 - eps,
        )
        ref_p_true = np.clip(
            _true_class_prob(train_data["ref_predicted_prob"].values, train_data["ref_true_label"].values),
            eps, 1 - eps,
        )
        train_data["log_ratio"] = np.log(target_p_true / ref_p_true)

    # Quantile binning for balanced bin sizes
    train_data["token_bin"] = pd.qcut(
        train_data["token_count"], q=n_bins, labels=False, duplicates="drop",
    )

    results = []
    for bin_id in sorted(train_data["token_bin"].unique()):
        bin_data = train_data[train_data["token_bin"] == bin_id]
        members = bin_data[bin_data["is_member"] == 1]
        nonmembers = bin_data[bin_data["is_member"] == 0]

        if len(members) < min_per_group or len(nonmembers) < min_per_group:
            continue

        # Loss-based signal
        mem_signal = -members["loss"].values
        non_signal = -nonmembers["loss"].values
        metrics = compute_mi_metrics(mem_signal, non_signal)

        entry = {
            "bin": int(bin_id),
            "token_count_min": int(bin_data["token_count"].min()),
            "token_count_max": int(bin_data["token_count"].max()),
            "token_count_mean": float(bin_data["token_count"].mean()),
            "n_members": len(members),
            "n_nonmembers": len(nonmembers),
            **metrics,
        }

        # Reference-model signal per bin
        if has_ref and "log_ratio" in bin_data.columns:
            ref_mem = members["log_ratio"].values
            ref_non = nonmembers["log_ratio"].values
            ref_metrics = compute_mi_metrics(ref_mem, ref_non)
            entry["ref_mi_auc"] = ref_metrics["mi_auc"]
            entry["ref_tpr_at_fpr_1pct"] = ref_metrics["tpr_at_fpr_1pct"]
            entry["ref_tpr_at_fpr_5pct"] = ref_metrics["tpr_at_fpr_5pct"]

        results.append(entry)

    return results


def breakdown_by_emotion_intensity(
    scores_df: pd.DataFrame,
    classifier_name: str,
    min_per_group: int = 15,
) -> list[dict]:
    """Compute MI-AUC per emotion-intensity bin (low / medium / high).

    Only meaningful for emotional-support data that contains an
    ``emotion_intensity`` column (sourced from ESConv).  Bins are defined
    in ``config.ESCONV_EMOTION_INTENSITY_BINS``.
    """
    if "emotion_intensity" not in scores_df.columns:
        return []

    _, split = config.parse_classifier_name(classifier_name)
    other_split = "B" if split == "A" else "A"

    clf_scores = scores_df[scores_df["classifier"] == classifier_name].copy()
    train_data = clf_scores[clf_scores["split"].isin(
        [f"{split}_train", f"{other_split}_train"]
    )].copy()
    train_data["is_member"] = (train_data["split"] == f"{split}_train").astype(int)

    # Filter to rows that actually have intensity (ESConv only; -1 = unknown)
    train_data = train_data[train_data["emotion_intensity"] > 0].copy()
    if train_data.empty:
        return []

    results = []
    for bin_name, (lo, hi) in config.ESCONV_EMOTION_INTENSITY_BINS.items():
        bin_data = train_data[
            (train_data["emotion_intensity"] >= lo) &
            (train_data["emotion_intensity"] <= hi)
        ]
        members = bin_data[bin_data["is_member"] == 1]
        nonmembers = bin_data[bin_data["is_member"] == 0]

        if len(members) < min_per_group or len(nonmembers) < min_per_group:
            continue

        mem_signal = -members["loss"].values
        non_signal = -nonmembers["loss"].values
        metrics = compute_mi_metrics(mem_signal, non_signal)

        results.append({
            "bin": bin_name,
            "intensity_range": f"{lo}-{hi}",
            "n_members": len(members),
            "n_nonmembers": len(nonmembers),
            **metrics,
        })

    return results


def breakdown_by_problem_type(
    scores_df: pd.DataFrame,
    classifier_name: str,
    min_per_group: int = 15,
) -> list[dict]:
    """Compute MI-AUC per ESConv problem type (e.g. depression, job crisis).

    Only meaningful for emotional-support data that contains a
    ``problem_type`` column.
    """
    if "problem_type" not in scores_df.columns:
        return []

    _, split = config.parse_classifier_name(classifier_name)
    other_split = "B" if split == "A" else "A"

    clf_scores = scores_df[scores_df["classifier"] == classifier_name].copy()
    train_data = clf_scores[clf_scores["split"].isin(
        [f"{split}_train", f"{other_split}_train"]
    )].copy()
    train_data["is_member"] = (train_data["split"] == f"{split}_train").astype(int)

    # Filter to rows that actually have a problem type (not 'none' from WildChat)
    train_data = train_data[
        ~train_data["problem_type"].isin(["none", "unknown", ""])
    ].copy()
    if train_data.empty:
        return []

    results = []
    for ptype, group in train_data.groupby("problem_type"):
        members = group[group["is_member"] == 1]
        nonmembers = group[group["is_member"] == 0]

        if len(members) < min_per_group or len(nonmembers) < min_per_group:
            continue

        mem_signal = -members["loss"].values
        non_signal = -nonmembers["loss"].values
        metrics = compute_mi_metrics(mem_signal, non_signal)

        results.append({
            "problem_type": str(ptype),
            "n_members": len(members),
            "n_nonmembers": len(nonmembers),
            **metrics,
        })

    results.sort(key=lambda r: r.get("mi_auc", 0.5), reverse=True)
    return results


# =========================================================================== #
# Logit-vector MIA attack
# =========================================================================== #

def _build_logit_vector_features(
    full_probs: np.ndarray,
    true_labels: np.ndarray,
) -> np.ndarray:
    """Build feature matrix for the logit-vector attack.

    For each example, construct:
      [softmax_0, ..., softmax_{K-1}, entropy, max_prob,
       true_class_rank, margin_1st_2nd]

    Parameters
    ----------
    full_probs : (N, K) softmax probability vectors
    true_labels : (N,) integer labels

    Returns
    -------
    (N, K+4) feature matrix
    """
    K = full_probs.shape[1]
    N = full_probs.shape[0]

    # Entropy: -sum(p * log(p))
    eps = 1e-12
    entropy = -np.sum(full_probs * np.log(full_probs + eps), axis=1)  # (N,)

    # Max probability
    max_prob = np.max(full_probs, axis=1)  # (N,)

    # True-class rank (1 = highest prob, K = lowest)
    # Sort descending, find where true class falls
    sorted_indices = np.argsort(-full_probs, axis=1)  # (N, K) descending
    true_class_rank = np.zeros(N)
    for i in range(N):
        rank = np.where(sorted_indices[i] == true_labels[i])[0]
        true_class_rank[i] = rank[0] + 1 if len(rank) > 0 else K

    # Margin between 1st and 2nd highest probabilities
    sorted_probs = np.sort(full_probs, axis=1)[:, ::-1]  # descending
    margin = sorted_probs[:, 0] - sorted_probs[:, 1] if K > 1 else sorted_probs[:, 0]

    features = np.column_stack([
        full_probs,        # K features
        entropy,           # 1
        max_prob,          # 1
        true_class_rank,   # 1
        margin,            # 1
    ])
    return features


def attack_logit_vector(
    scores_df: pd.DataFrame,
    classifier_name: str,
    num_labels: int = 2,
) -> dict | None:
    """Logit-vector MIA: train a learned membership classifier on the full
    softmax distribution from shadow model outputs.

    Uses the paired classifier (same model size, other split) as the
    shadow model.  Constructs a feature matrix from full probability
    vectors + derived features, trains LogisticRegression on the shadow
    model's IN/OUT split, and evaluates on the target model.

    Returns attack result dict, or None if prerequisites are missing.
    """
    model_size, split = config.parse_classifier_name(classifier_name)
    other_split = "B" if split == "A" else "A"
    shadow_classifier = f"{model_size}_{other_split}"

    # Load full probability vectors
    target_probs_path = config.SCORES_DIR / f"full_probs_{classifier_name}.npy"
    shadow_probs_path = config.SCORES_DIR / f"full_probs_{shadow_classifier}.npy"

    if not target_probs_path.exists() or not shadow_probs_path.exists():
        print(f"    Skipping logit-vector attack: missing full_probs for "
              f"{classifier_name} or {shadow_classifier}")
        return None

    target_full_probs = np.load(target_probs_path)
    shadow_full_probs = np.load(shadow_probs_path)

    # Get score DataFrames for both classifiers
    target_scores = scores_df[scores_df["classifier"] == classifier_name].copy()
    shadow_scores = scores_df[scores_df["classifier"] == shadow_classifier].copy()

    # Filter to training splits only
    target_train = target_scores[target_scores["split"].isin(
        [f"{split}_train", f"{other_split}_train"]
    )].reset_index(drop=True)

    shadow_train = shadow_scores[shadow_scores["split"].isin(
        [f"{split}_train", f"{other_split}_train"]
    )].reset_index(drop=True)

    if len(target_train) == 0 or len(shadow_train) == 0:
        return None

    # Need to align full_probs with the filtered training data
    # full_probs was saved in the same order as the scored splits
    # (A_train, B_train, attack_cal, attack_eval)
    splits_to_score = ["A_train", "B_train", "attack_cal", "attack_eval"]

    # Get indices of training split rows within the full scored data
    target_all = scores_df[scores_df["classifier"] == classifier_name].copy()
    target_all = target_all[target_all["split"].isin(splits_to_score)].reset_index(drop=True)
    target_train_mask = target_all["split"].isin(
        [f"{split}_train", f"{other_split}_train"]
    ).values

    shadow_all = scores_df[scores_df["classifier"] == shadow_classifier].copy()
    shadow_all = shadow_all[shadow_all["split"].isin(splits_to_score)].reset_index(drop=True)
    shadow_train_mask = shadow_all["split"].isin(
        [f"{split}_train", f"{other_split}_train"]
    ).values

    # Verify dimensions match
    if len(target_all) != target_full_probs.shape[0]:
        print(f"    Logit-vector: shape mismatch target "
              f"({len(target_all)} vs {target_full_probs.shape[0]})")
        return None
    if len(shadow_all) != shadow_full_probs.shape[0]:
        print(f"    Logit-vector: shape mismatch shadow "
              f"({len(shadow_all)} vs {shadow_full_probs.shape[0]})")
        return None

    # Extract training-split probs
    target_train_probs = target_full_probs[target_train_mask]
    shadow_train_probs = shadow_full_probs[shadow_train_mask]

    target_train_labels = target_all.loc[target_train_mask, "true_label"].values
    shadow_train_labels = shadow_all.loc[shadow_train_mask, "true_label"].values
    target_train_splits = target_all.loc[target_train_mask, "split"].values
    shadow_train_splits = shadow_all.loc[shadow_train_mask, "split"].values

    # Build features
    target_features = _build_logit_vector_features(target_train_probs, target_train_labels)
    shadow_features = _build_logit_vector_features(shadow_train_probs, shadow_train_labels)

    # Shadow model membership labels:
    # For the shadow classifier (trained on other_split_train),
    # other_split_train = members, split_train = non-members
    shadow_membership = (shadow_train_splits == f"{other_split}_train").astype(int)

    # Target model membership labels:
    # For the target classifier (trained on split_train),
    # split_train = members, other_split_train = non-members
    target_membership = (target_train_splits == f"{split}_train").astype(int)

    # Train logistic regression on shadow data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_shadow = scaler.fit_transform(shadow_features)
    X_target = scaler.transform(target_features)

    lr = LogisticRegression(max_iter=1000, random_state=config.SEED)
    lr.fit(X_shadow, shadow_membership)

    # Predict on target
    target_pred_proba = lr.predict_proba(X_target)[:, 1]

    # Evaluate
    member_signal = target_pred_proba[target_membership == 1]
    nonmember_signal = target_pred_proba[target_membership == 0]

    metrics = compute_mi_metrics(member_signal, nonmember_signal)
    cis = bootstrap_ci(member_signal, nonmember_signal)

    # Feature importances (logistic regression coefficients)
    feature_names = (
        [f"softmax_{i}" for i in range(num_labels)]
        + ["entropy", "max_prob", "true_class_rank", "margin_1st_2nd"]
    )
    coefficients = {
        name: float(coef)
        for name, coef in zip(feature_names, lr.coef_[0])
    }

    return {
        "attack": "logit_vector",
        "classifier": classifier_name,
        "shadow_classifier": shadow_classifier,
        "num_labels": num_labels,
        "n_features": len(feature_names),
        "n_members": int(target_membership.sum()),
        "n_nonmembers": int((1 - target_membership).sum()),
        **metrics,
        "confidence_intervals": cis,
        "feature_importances": coefficients,
    }


# =========================================================================== #
# Canary probing attack (spurious label memorization)
# =========================================================================== #

@torch.no_grad()
def score_canary_probes(
    model,
    tokenizer,
    canary_metadata: dict,
    batch_size: int = config.INFERENCE_BATCH_SIZE,
    max_length: int = config.MAX_SEQ_LEN,
) -> pd.DataFrame:
    """Score each unique canary by feeding probe text to the classifier.

    For each canary ``(name, number)`` we construct a probe string
    using ``CANARY_TEMPLATE_PROBE`` (which omits the number, testing
    whether the model generalises beyond exact-match) and record both
    the legacy ``p_unsafe`` (P(class 1)) and the multiclass-aware
    ``p_assigned_label`` — P(canary's assigned label) — which is the
    primary signal for category-targeted canary attacks.

    Returns a DataFrame with one row per unique canary.
    """
    canaries = canary_metadata["canaries"]
    num_labels = canary_metadata.get("num_labels", 2)
    probe_texts = []
    for c in canaries:
        probe = config.CANARY_TEMPLATE_PROBE.format(name=c["name"])
        formatted = format_for_classification(probe, tokenizer, task="safety")
        probe_texts.append(formatted)

    all_full_probs: list[np.ndarray] = []

    n = len(probe_texts)
    for start in tqdm(range(0, n, batch_size), desc="  Scoring canary probes"):
        end = min(start + batch_size, n)
        batch = probe_texts[start:end]

        encodings = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            max_length=max_length,
            return_tensors="pt",
        )

        device = next(model.parameters()).device
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.float()
        probs = F.softmax(logits, dim=-1)  # (batch, num_labels)

        all_full_probs.append(probs.cpu().numpy())

    full_probs = np.vstack(all_full_probs)  # (n_canaries, num_labels)

    rows = []
    for i, c in enumerate(canaries):
        canary_label = c.get("canary_label", 1)  # backward compat: default→1

        row = {
            "canary_id": c["canary_id"],
            "name": c["name"],
            "number": c["number"],
            "canary_label": int(canary_label),
            "source_split": c["split"],
            "paired_labels": c["paired_labels"],
            "n_examples": len(c["example_indices"]),
            # Primary multiclass-aware signal: P(canary's assigned label)
            "p_assigned_label": float(full_probs[i, canary_label]),
            # Legacy binary signals (backward compat)
            "p_unsafe": float(full_probs[i, 1]) if num_labels >= 2 else 0.0,
            "logit_0": float(full_probs[i, 0]),
            "logit_1": float(full_probs[i, 1]) if num_labels >= 2 else 0.0,
        }

        # Store full probability vector per canary for reference-model attacks
        for lbl_idx in range(full_probs.shape[1]):
            row[f"p_class_{lbl_idx}"] = float(full_probs[i, lbl_idx])

        rows.append(row)

    return pd.DataFrame(rows)


def run_canary_attack(
    classifier_name: str,
    probe_scores: pd.DataFrame,
    num_labels: int = 2,
) -> list[dict]:
    """Run canary membership inference: can the model distinguish member
    vs non-member canaries from probe-only input?

    For classifier ``X_A``, member canaries are those inserted into
    ``A_train``; non-member canaries are those inserted into ``B_train``
    (which ``X_A`` never saw).

    Multiple signal variants are tested:
      - ``p_assigned_label``: P(canary's assigned label) — primary signal,
        works for both binary and multiclass classifiers
      - ``p_unsafe``: raw P(class 1) — legacy binary signal
      - ``confidence``: max(P(assigned), 1-P(assigned)) — magnitude
      - ``deviation``: |P(assigned) − baseline_mean| — calibrated

    When *num_labels* > 2, per-category breakdowns are also computed so
    that each harm category's memorisation leakage can be measured
    independently.

    Returns a list of result dicts (one per signal variant).
    """
    _, split = config.parse_classifier_name(classifier_name)
    member_split = f"{split}_train"
    other_split = "B" if split == "A" else "A"
    nonmember_split = f"{other_split}_train"

    members = probe_scores[probe_scores["source_split"] == member_split]
    nonmembers = probe_scores[probe_scores["source_split"] == nonmember_split]

    if len(members) < 5 or len(nonmembers) < 5:
        print(f"    Skipping canary attack for {classifier_name}: "
              f"too few probes (mem={len(members)}, non={len(nonmembers)})")
        return []

    results = []

    # ---- Primary signal: P(assigned label) ----
    # This works for both binary (canary_label ∈ {0,1}) and multiclass.
    has_p_assigned = "p_assigned_label" in members.columns
    if has_p_assigned:
        mem_pa = members["p_assigned_label"].values
        non_pa = nonmembers["p_assigned_label"].values
        metrics = compute_mi_metrics(mem_pa, non_pa)
        cis = bootstrap_ci(mem_pa, non_pa)
        results.append({
            "attack": "canary_p_assigned_label",
            "classifier": classifier_name,
            "signal": "p_assigned_label",
            "n_member_canaries": len(members),
            "n_nonmember_canaries": len(nonmembers),
            **metrics,
            "confidence_intervals": cis,
        })

    # ---- Legacy signal: raw P(unsafe) (class 1) ----
    if "p_unsafe" in members.columns:
        mem_p = members["p_unsafe"].values
        non_p = nonmembers["p_unsafe"].values
        metrics = compute_mi_metrics(mem_p, non_p)
        cis = bootstrap_ci(mem_p, non_p)
        results.append({
            "attack": "canary_p_unsafe",
            "classifier": classifier_name,
            "signal": "p_unsafe",
            "n_member_canaries": len(members),
            "n_nonmember_canaries": len(nonmembers),
            **metrics,
            "confidence_intervals": cis,
        })

    # ---- Confidence signal ----
    # Use p_assigned_label when available; fall back to p_unsafe
    sig_col = "p_assigned_label" if has_p_assigned else "p_unsafe"
    mem_sig_raw = members[sig_col].values
    non_sig_raw = nonmembers[sig_col].values

    mem_conf = np.maximum(mem_sig_raw, 1.0 - mem_sig_raw)
    non_conf = np.maximum(non_sig_raw, 1.0 - non_sig_raw)
    metrics = compute_mi_metrics(mem_conf, non_conf)
    cis = bootstrap_ci(mem_conf, non_conf)
    results.append({
        "attack": "canary_confidence",
        "classifier": classifier_name,
        "signal": "confidence",
        "n_member_canaries": len(members),
        "n_nonmember_canaries": len(nonmembers),
        **metrics,
        "confidence_intervals": cis,
    })

    # ---- Deviation signal ----
    baseline_mean = non_sig_raw.mean()
    mem_dev = np.abs(mem_sig_raw - baseline_mean)
    non_dev = np.abs(non_sig_raw - baseline_mean)
    metrics = compute_mi_metrics(mem_dev, non_dev)
    cis = bootstrap_ci(mem_dev, non_dev)
    results.append({
        "attack": "canary_deviation",
        "classifier": classifier_name,
        "signal": "deviation",
        "n_member_canaries": len(members),
        "n_nonmember_canaries": len(nonmembers),
        **metrics,
        "confidence_intervals": cis,
    })

    # ---- Subgroup: break down by paired training label ----
    # Determine label name map — use multiclass names when available
    if num_labels > 2:
        label_name_map = config.get_label_names("beavertails", "multiclass")
    else:
        label_name_map = {0: "safe", 1: "unsafe"}

    for label_val, label_name in label_name_map.items():
        # Filter canaries whose training examples had this label
        mem_sub = members[members["paired_labels"].apply(
            lambda pl, lv=label_val: any(l == lv for l in pl)
        )]
        non_sub = nonmembers[nonmembers["paired_labels"].apply(
            lambda pl, lv=label_val: any(l == lv for l in pl)
        )]
        if len(mem_sub) < 5 or len(non_sub) < 5:
            continue
        mem_sig = mem_sub[sig_col].values
        non_sig = non_sub[sig_col].values
        metrics = compute_mi_metrics(mem_sig, non_sig)
        cis = bootstrap_ci(mem_sig, non_sig)
        results.append({
            "attack": f"canary_p_assigned_label_{label_name}",
            "classifier": classifier_name,
            "signal": sig_col,
            "subgroup": f"paired_label={label_name}",
            "n_member_canaries": len(mem_sub),
            "n_nonmember_canaries": len(non_sub),
            **metrics,
            "confidence_intervals": cis,
        })

    # ---- Per-category breakdown by canary_label (multiclass) ----
    if num_labels > 2 and "canary_label" in probe_scores.columns:
        for label_id in sorted(probe_scores["canary_label"].unique()):
            subset = probe_scores[probe_scores["canary_label"] == label_id]
            mem_sub = subset[subset["source_split"] == member_split]
            non_sub = subset[subset["source_split"] == nonmember_split]
            if len(mem_sub) < 3 or len(non_sub) < 3:
                continue
            mem_sig = mem_sub["p_assigned_label"].values
            non_sig = non_sub["p_assigned_label"].values
            metrics = compute_mi_metrics(mem_sig, non_sig)
            lbl_name = label_name_map.get(int(label_id), f"class_{label_id}")
            results.append({
                "attack": f"canary_category_{lbl_name}",
                "classifier": classifier_name,
                "signal": "p_assigned_label",
                "subgroup": f"canary_label={lbl_name}",
                "n_member_canaries": len(mem_sub),
                "n_nonmember_canaries": len(non_sub),
                **metrics,
            })

    return results


def run_canary_reference_model_attack(
    classifier_name: str,
    probe_scores_target: pd.DataFrame,
    probe_scores_reference: pd.DataFrame,
) -> dict | None:
    """LiRA-style canary attack using the paired classifier as reference.

    Uses the multiclass-aware ``p_assigned_label`` when available, with
    fallback to the legacy ``p_unsafe`` signal.

    Signal = log(P_target(assigned_label) / P_reference(assigned_label)).

    Both DataFrames must have been scored on the same canary set (aligned
    by canary_id).
    """
    model_size, split = config.parse_classifier_name(classifier_name)
    other_split = "B" if split == "A" else "A"
    member_split = f"{split}_train"
    nonmember_split = f"{other_split}_train"

    # Align on canary_id
    merged = probe_scores_target.merge(
        probe_scores_reference,
        on="canary_id",
        suffixes=("_target", "_ref"),
    )
    if len(merged) < 10:
        return None

    # Choose signal column: prefer p_assigned_label (multiclass-aware)
    eps = 1e-7
    if "p_assigned_label_target" in merged.columns:
        p_target = np.clip(merged["p_assigned_label_target"].values, eps, 1 - eps)
        p_ref = np.clip(merged["p_assigned_label_ref"].values, eps, 1 - eps)
        signal_name = "log_ratio_p_assigned_label"
    else:
        p_target = np.clip(merged["p_unsafe_target"].values, eps, 1 - eps)
        p_ref = np.clip(merged["p_unsafe_ref"].values, eps, 1 - eps)
        signal_name = "log_ratio_p_unsafe"

    log_ratio = np.log(p_target / p_ref)

    # Determine membership from source_split_target
    member_mask = merged["source_split_target"] == member_split
    nonmember_mask = merged["source_split_target"] == nonmember_split

    if member_mask.sum() < 5 or nonmember_mask.sum() < 5:
        return None

    member_signal = log_ratio[member_mask]
    nonmember_signal = log_ratio[nonmember_mask]

    metrics = compute_mi_metrics(member_signal, nonmember_signal)
    cis = bootstrap_ci(member_signal, nonmember_signal)

    ref_classifier = f"{model_size}_{other_split}"
    return {
        "attack": "canary_reference_model",
        "classifier": classifier_name,
        "reference_classifier": ref_classifier,
        "signal": signal_name,
        "n_member_canaries": int(member_mask.sum()),
        "n_nonmember_canaries": int(nonmember_mask.sum()),
        **metrics,
        "confidence_intervals": cis,
    }


# =========================================================================== #
# Canary attribute inference attack
# =========================================================================== #

def canary_attribute_inference_attack(
    classifier_name: str,
    probe_scores: pd.DataFrame,
    num_labels: int = 2,
    n_bootstrap: int = 1000,
) -> dict:
    """Attribute inference attack via canary probes.

    Tests whether querying the classifier with *only* a canary name
    reveals the **private label** that name was paired with during
    training.  This is distinct from membership inference: MIA asks
    "was this name in the training set?", while AIA asks "what label
    was this name associated with?"

    For classifier ``X_A``:

    * **Member canaries** (inserted into ``A_train``): the classifier
      saw ``"Hi, I'm Alice - 3847. "`` prepended to training texts
      with a specific label.  If the classifier memorised the
      name→label association, probing with ``"Hi, I'm Alice. "`` should
      yield argmax = canary_label more often than chance.

    * **Non-member canaries** (inserted into ``B_train``): the
      classifier never saw these names.  They serve as a **control**
      — any label prediction accuracy here is due to content priors
      (e.g. benign probe text → safe), not memorisation.

    Metrics computed:

    * ``member_label_accuracy``: fraction of member canaries where
      argmax(Pr) == canary_label
    * ``nonmember_label_accuracy``: same for non-members (baseline)
    * ``chance_accuracy``: 1 / num_labels
    * ``attacker_advantage``: member - nonmember accuracy (the
      privacy-relevant quantity; positive = memorisation-driven leak)
    * ``advantage_over_chance``: member - chance accuracy
    * ``mean_p_assigned_member`` / ``mean_p_assigned_nonmember``:
      average P(canary_label) for each group
    * ``p_assigned_advantage``: difference of the above (continuous
      signal, more sensitive than argmax accuracy)
    * Bootstrap 95% CIs on attacker_advantage and p_assigned_advantage
    * Per-label breakdown (when num_labels > 2)

    Returns a single result dict.
    """
    _, split = config.parse_classifier_name(classifier_name)
    member_split = f"{split}_train"
    other_split = "B" if split == "A" else "A"
    nonmember_split = f"{other_split}_train"

    members = probe_scores[probe_scores["source_split"] == member_split]
    nonmembers = probe_scores[probe_scores["source_split"] == nonmember_split]

    if len(members) < 5 or len(nonmembers) < 5:
        return {}

    # --- Argmax label prediction ---
    p_class_cols = sorted(
        [c for c in probe_scores.columns if c.startswith("p_class_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    if not p_class_cols:
        return {}

    mem_probs = members[p_class_cols].values  # (n_mem, K)
    non_probs = nonmembers[p_class_cols].values  # (n_non, K)
    mem_predicted = np.argmax(mem_probs, axis=1)
    non_predicted = np.argmax(non_probs, axis=1)
    mem_canary_labels = members["canary_label"].values
    non_canary_labels = nonmembers["canary_label"].values

    mem_correct = (mem_predicted == mem_canary_labels).astype(float)
    non_correct = (non_predicted == non_canary_labels).astype(float)

    mem_acc = float(mem_correct.mean())
    non_acc = float(non_correct.mean())
    chance = 1.0 / num_labels
    advantage = mem_acc - non_acc

    # --- P(assigned label) continuous signal ---
    mem_p_assigned = members["p_assigned_label"].values
    non_p_assigned = nonmembers["p_assigned_label"].values
    p_assigned_adv = float(mem_p_assigned.mean() - non_p_assigned.mean())

    # --- Bootstrap CIs ---
    rng = np.random.RandomState(config.SEED)
    boot_advantages = np.empty(n_bootstrap)
    boot_p_assigned_adv = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        m_idx = rng.choice(len(mem_correct), size=len(mem_correct), replace=True)
        n_idx = rng.choice(len(non_correct), size=len(non_correct), replace=True)
        boot_advantages[b] = mem_correct[m_idx].mean() - non_correct[n_idx].mean()
        boot_p_assigned_adv[b] = (
            mem_p_assigned[m_idx].mean() - non_p_assigned[n_idx].mean()
        )

    adv_ci = (
        float(np.percentile(boot_advantages, 2.5)),
        float(np.percentile(boot_advantages, 97.5)),
    )
    p_adv_ci = (
        float(np.percentile(boot_p_assigned_adv, 2.5)),
        float(np.percentile(boot_p_assigned_adv, 97.5)),
    )

    result: dict = {
        "attack": "canary_attribute_inference",
        "classifier": classifier_name,
        "num_labels": num_labels,
        "n_member_canaries": len(members),
        "n_nonmember_canaries": len(nonmembers),
        "chance_accuracy": round(chance, 4),
        "member_label_accuracy": round(mem_acc, 4),
        "nonmember_label_accuracy": round(non_acc, 4),
        "attacker_advantage": round(advantage, 4),
        "advantage_over_chance": round(mem_acc - chance, 4),
        "advantage_ci_95": [round(adv_ci[0], 4), round(adv_ci[1], 4)],
        "mean_p_assigned_member": round(float(mem_p_assigned.mean()), 4),
        "mean_p_assigned_nonmember": round(float(non_p_assigned.mean()), 4),
        "p_assigned_advantage": round(p_assigned_adv, 4),
        "p_assigned_advantage_ci_95": [round(p_adv_ci[0], 4), round(p_adv_ci[1], 4)],
    }

    # --- Per-label breakdown ---
    label_breakdown = {}
    for lbl in sorted(probe_scores["canary_label"].unique()):
        m_sub = members[members["canary_label"] == lbl]
        n_sub = nonmembers[nonmembers["canary_label"] == lbl]
        if len(m_sub) < 2 or len(n_sub) < 2:
            continue
        m_pred = np.argmax(m_sub[p_class_cols].values, axis=1)
        n_pred = np.argmax(n_sub[p_class_cols].values, axis=1)
        m_acc_lbl = float((m_pred == lbl).mean())
        n_acc_lbl = float((n_pred == lbl).mean())
        m_pa = float(m_sub["p_assigned_label"].mean())
        n_pa = float(n_sub["p_assigned_label"].mean())

        label_name = str(lbl)
        if num_labels > 2:
            names = config.get_label_names("beavertails", "multiclass")
            label_name = names.get(int(lbl), f"class_{lbl}")
        elif int(lbl) == 0:
            label_name = "benign"
        elif int(lbl) == 1:
            label_name = "positive"

        label_breakdown[label_name] = {
            "n_member": len(m_sub),
            "n_nonmember": len(n_sub),
            "member_accuracy": round(m_acc_lbl, 4),
            "nonmember_accuracy": round(n_acc_lbl, 4),
            "advantage": round(m_acc_lbl - n_acc_lbl, 4),
            "mean_p_assigned_member": round(m_pa, 4),
            "mean_p_assigned_nonmember": round(n_pa, 4),
            "p_assigned_advantage": round(m_pa - n_pa, 4),
        }

    if label_breakdown:
        result["per_label"] = label_breakdown

    return result


# =========================================================================== #
# Boundary canary attack (decision-boundary memorization test)
# =========================================================================== #

@torch.no_grad()
def find_boundary_examples(
    model,
    tokenizer,
    df: pd.DataFrame,
    num_labels: int = 2,
    n_per_category: int = config.BOUNDARY_N_PER_CATEGORY,
    batch_size: int = config.INFERENCE_BATCH_SIZE,
    max_length: int = config.MAX_SEQ_LEN,
) -> pd.DataFrame:
    """Find training examples where the classifier is least confident.

    These sit on the decision boundary — the exact place where
    memorisation would matter most and be most detectable.

    Selects the *n_per_category* examples with the lowest P(true label)
    for each label in the training splits, ensuring balanced category
    coverage.

    Returns a DataFrame of boundary examples with added ``p_true_label``
    and ``entropy`` columns.
    """
    label_mode = "multiclass" if num_labels > 2 else "binary"
    train = df[df["split"].isin(["A_train", "B_train"])].copy().reset_index(drop=True)
    texts = train["text"].tolist()
    labels = train["label"].tolist()

    # Score every training example
    all_full_probs: list[np.ndarray] = []
    n = len(texts)
    for start in tqdm(range(0, n, batch_size), desc="  Finding boundary examples"):
        end = min(start + batch_size, n)
        batch_formatted = [
            format_for_classification(t, tokenizer, task="safety",
                                      label_mode=label_mode)
            for t in texts[start:end]
        ]

        encodings = tokenizer(
            batch_formatted,
            truncation=True,
            padding="longest",
            max_length=max_length,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        logits = model(input_ids=input_ids,
                       attention_mask=attention_mask).logits.float()
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        all_full_probs.append(probs)

    all_probs = np.vstack(all_full_probs)  # (N, num_labels)

    # Compute P(true label) and entropy for each example
    p_true = np.array([all_probs[i, labels[i]] for i in range(len(labels))])
    entropy = -np.sum(all_probs * np.log(all_probs + 1e-10), axis=1)

    train["p_true_label"] = p_true
    train["entropy"] = entropy

    # Store the full probability vector for potential later use
    for c in range(all_probs.shape[1]):
        train[f"p_class_{c}"] = all_probs[:, c]

    # Select the most uncertain examples per category
    boundary_parts = []
    for lbl in sorted(train["label"].unique()):
        cat_df = train[train["label"] == lbl].copy()
        # Sort by P(true label) ascending — least confident first
        cat_df = cat_df.sort_values("p_true_label", ascending=True)
        selected = cat_df.head(n_per_category)
        boundary_parts.append(selected)

        if len(selected) > 0:
            label_name = config.BEAVERTAILS_MULTICLASS_LABEL_NAMES.get(
                lbl, f"class_{lbl}"
            ) if num_labels > 2 else ("safe" if lbl == 0 else "unsafe")
            print(f"    Label {lbl:2d} ({label_name:45s}): "
                  f"n={len(selected):3d}, "
                  f"mean P(true)={selected['p_true_label'].mean():.4f}, "
                  f"mean entropy={selected['entropy'].mean():.3f}")

    boundary_df = pd.concat(boundary_parts, ignore_index=True)
    print(f"  Selected {len(boundary_df)} boundary examples total")
    return boundary_df


@torch.no_grad()
def score_boundary_probes(
    model,
    tokenizer,
    boundary_df: pd.DataFrame,
    num_labels: int = 2,
    batch_size: int = config.INFERENCE_BATCH_SIZE,
    max_length: int = config.MAX_SEQ_LEN,
) -> pd.DataFrame:
    """Score boundary examples through a classifier.

    For each boundary example, records:
      - ``p_assigned_label``: P(original true label) — the matched signal
      - ``p_class_*``: full probability vector
      - ``predicted_class``: argmax prediction
      - ``entropy``: prediction entropy

    This lets us test whether the model's confidence on boundary examples
    differs between member and non-member texts.
    """
    label_mode = "multiclass" if num_labels > 2 else "binary"
    texts = boundary_df["text"].tolist()

    all_full_probs: list[np.ndarray] = []
    n = len(texts)
    for start in tqdm(range(0, n, batch_size), desc="  Scoring boundary probes"):
        end = min(start + batch_size, n)
        batch_formatted = [
            format_for_classification(t, tokenizer, task="safety",
                                      label_mode=label_mode)
            for t in texts[start:end]
        ]

        encodings = tokenizer(
            batch_formatted,
            truncation=True,
            padding="longest",
            max_length=max_length,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        logits = model(input_ids=input_ids,
                       attention_mask=attention_mask).logits.float()
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        all_full_probs.append(probs)

    full_probs = np.vstack(all_full_probs)

    rows = []
    for i in range(len(boundary_df)):
        row_data = boundary_df.iloc[i]
        true_label = int(row_data["label"])
        probs_i = full_probs[i]

        row = {
            "boundary_idx": i,
            "text_len": len(row_data["text"]),
            "true_label": true_label,
            "source_split": row_data["split"],
            "p_true_label": float(probs_i[true_label]),
            "predicted_class": int(np.argmax(probs_i)),
            "max_prob": float(np.max(probs_i)),
            "entropy": float(-np.sum(probs_i * np.log(probs_i + 1e-10))),
            "p_true_at_selection": float(row_data["p_true_label"]),
            "entropy_at_selection": float(row_data["entropy"]),
        }

        # Full probability vector
        for c in range(full_probs.shape[1]):
            row[f"p_class_{c}"] = float(probs_i[c])

        rows.append(row)

    return pd.DataFrame(rows)


def run_boundary_canary_attack(
    classifier_name: str,
    boundary_scores: pd.DataFrame,
    num_labels: int = 2,
) -> list[dict]:
    """Boundary canary MIA: does the classifier leak membership for
    decision-boundary examples?

    For classifier ``X_A``, members are boundary examples from ``A_train``;
    non-members are boundary examples from ``B_train``.  We test multiple
    signals:

      - ``p_true_label``: P(example's true label) — confidence signal.
        Higher for members if the model memorised margin examples.
      - ``entropy``: prediction entropy — lower for memorised examples.
      - ``loss``: cross-entropy loss — lower for memorised examples.

    Unlike the original canary probes (which used synthetic benign text),
    boundary probes use *real training text* that the classifier finds
    ambiguous.  This isolates memorisation from content-based classification.

    Returns a list of result dicts (one per signal variant + per-category).
    """
    _, split = config.parse_classifier_name(classifier_name)
    member_split = f"{split}_train"
    other_split = "B" if split == "A" else "A"
    nonmember_split = f"{other_split}_train"

    members = boundary_scores[boundary_scores["source_split"] == member_split]
    nonmembers = boundary_scores[boundary_scores["source_split"] == nonmember_split]

    if len(members) < 5 or len(nonmembers) < 5:
        print(f"    Skipping boundary attack for {classifier_name}: "
              f"too few probes (mem={len(members)}, non={len(nonmembers)})")
        return []

    results = []

    # ---- Signal 1: P(true label) — confidence on correct class ----
    mem_p = members["p_true_label"].values
    non_p = nonmembers["p_true_label"].values
    metrics = compute_mi_metrics(mem_p, non_p)
    cis = bootstrap_ci(mem_p, non_p)
    results.append({
        "attack": "boundary_p_true_label",
        "classifier": classifier_name,
        "signal": "p_true_label",
        "n_member": len(members),
        "n_nonmember": len(nonmembers),
        **metrics,
        "confidence_intervals": cis,
    })

    # ---- Signal 2: Entropy (lower = more confident = more memorised) ----
    # Negate entropy so higher = more memorised (matches MI-AUC convention)
    mem_ent = -members["entropy"].values
    non_ent = -nonmembers["entropy"].values
    metrics = compute_mi_metrics(mem_ent, non_ent)
    cis = bootstrap_ci(mem_ent, non_ent)
    results.append({
        "attack": "boundary_neg_entropy",
        "classifier": classifier_name,
        "signal": "neg_entropy",
        "n_member": len(members),
        "n_nonmember": len(nonmembers),
        **metrics,
        "confidence_intervals": cis,
    })

    # ---- Signal 3: Max probability (margin confidence) ----
    mem_max = members["max_prob"].values
    non_max = nonmembers["max_prob"].values
    metrics = compute_mi_metrics(mem_max, non_max)
    cis = bootstrap_ci(mem_max, non_max)
    results.append({
        "attack": "boundary_max_prob",
        "classifier": classifier_name,
        "signal": "max_prob",
        "n_member": len(members),
        "n_nonmember": len(nonmembers),
        **metrics,
        "confidence_intervals": cis,
    })

    # ---- Per-category breakdown ----
    label_names = config.get_label_names("beavertails",
                                         "multiclass" if num_labels > 2 else "binary")
    for lbl in sorted(boundary_scores["true_label"].unique()):
        subset = boundary_scores[boundary_scores["true_label"] == lbl]
        mem_sub = subset[subset["source_split"] == member_split]
        non_sub = subset[subset["source_split"] == nonmember_split]
        if len(mem_sub) < 3 or len(non_sub) < 3:
            continue

        mem_sig = mem_sub["p_true_label"].values
        non_sig = non_sub["p_true_label"].values
        metrics = compute_mi_metrics(mem_sig, non_sig)
        lbl_name = label_names.get(int(lbl), f"class_{lbl}")
        results.append({
            "attack": f"boundary_category_{lbl_name}",
            "classifier": classifier_name,
            "signal": "p_true_label",
            "subgroup": f"true_label={lbl_name}",
            "n_member": len(mem_sub),
            "n_nonmember": len(non_sub),
            **metrics,
        })

    return results


def run_boundary_reference_model_attack(
    classifier_name: str,
    boundary_scores_target: pd.DataFrame,
    boundary_scores_reference: pd.DataFrame,
) -> dict | None:
    """LiRA-style boundary attack using the paired classifier as reference.

    Signal = log(P_target(true_label) / P_reference(true_label)).

    If the target memorised boundary examples, the ratio is higher for
    members than non-members — the reference model controls for example
    difficulty.
    """
    model_size, split = config.parse_classifier_name(classifier_name)
    other_split = "B" if split == "A" else "A"
    member_split = f"{split}_train"
    nonmember_split = f"{other_split}_train"

    # Align by boundary_idx
    merged = boundary_scores_target.merge(
        boundary_scores_reference,
        on="boundary_idx",
        suffixes=("_target", "_ref"),
    )
    if len(merged) < 10:
        return None

    eps = 1e-7
    p_target = np.clip(merged["p_true_label_target"].values, eps, 1 - eps)
    p_ref = np.clip(merged["p_true_label_ref"].values, eps, 1 - eps)
    log_ratio = np.log(p_target / p_ref)

    member_mask = merged["source_split_target"] == member_split
    nonmember_mask = merged["source_split_target"] == nonmember_split

    if member_mask.sum() < 5 or nonmember_mask.sum() < 5:
        return None

    member_signal = log_ratio[member_mask]
    nonmember_signal = log_ratio[nonmember_mask]

    metrics = compute_mi_metrics(member_signal, nonmember_signal)
    cis = bootstrap_ci(member_signal, nonmember_signal)

    ref_classifier = f"{model_size}_{other_split}"
    return {
        "attack": "boundary_reference_model",
        "classifier": classifier_name,
        "reference_classifier": ref_classifier,
        "signal": "log_ratio_p_true_label",
        "n_member": int(member_mask.sum()),
        "n_nonmember": int(nonmember_mask.sum()),
        **metrics,
        "confidence_intervals": cis,
    }


def run_boundary_loss_attack(
    classifier_name: str,
    boundary_scores: pd.DataFrame,
) -> dict | None:
    """Loss-based boundary attack: use -log P(true label) as membership signal.

    Lower loss on boundary examples → model memorised them.
    We negate so that *higher* signal = more likely member (MI-AUC convention).
    """
    _, split = config.parse_classifier_name(classifier_name)
    member_split = f"{split}_train"
    other_split = "B" if split == "A" else "A"
    nonmember_split = f"{other_split}_train"

    members = boundary_scores[boundary_scores["source_split"] == member_split]
    nonmembers = boundary_scores[boundary_scores["source_split"] == nonmember_split]

    if len(members) < 5 or len(nonmembers) < 5:
        return None

    eps = 1e-7
    # Negate loss so higher = more memorised
    mem_signal = -np.log(np.clip(members["p_true_label"].values, eps, 1.0))
    non_signal = -np.log(np.clip(nonmembers["p_true_label"].values, eps, 1.0))
    # Lower loss for members → negate so higher = member
    mem_signal = -mem_signal
    non_signal = -non_signal

    metrics = compute_mi_metrics(mem_signal, non_signal)
    cis = bootstrap_ci(mem_signal, non_signal)

    return {
        "attack": "boundary_loss",
        "classifier": classifier_name,
        "signal": "neg_cross_entropy_loss",
        "n_member": len(members),
        "n_nonmember": len(nonmembers),
        **metrics,
        "confidence_intervals": cis,
    }


def _extract_boundary_full_probs(boundary_scores: pd.DataFrame) -> np.ndarray:
    """Extract the full probability matrix from boundary score columns.

    Boundary scores store per-class probabilities as ``p_class_0``,
    ``p_class_1``, etc.  Returns (N, K) array.
    """
    p_cols = sorted(
        [c for c in boundary_scores.columns if c.startswith("p_class_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    return boundary_scores[p_cols].values


def run_boundary_logit_vector_attack(
    classifier_name: str,
    boundary_scores_target: pd.DataFrame,
    boundary_scores_shadow: pd.DataFrame,
) -> dict | None:
    """Logit-vector boundary attack: train a learned membership classifier
    on boundary-example softmax distributions from the shadow (paired) model,
    then evaluate on the target model's boundary examples.

    Mirrors ``attack_logit_vector`` but restricted to boundary examples.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    model_size, split = config.parse_classifier_name(classifier_name)
    other_split = "B" if split == "A" else "A"
    member_split = f"{split}_train"
    nonmember_split = f"{other_split}_train"

    # --- Shadow (calibration) data: build features + labels ---
    shadow_mem = boundary_scores_shadow[
        boundary_scores_shadow["source_split"] == f"{other_split}_train"
    ]
    shadow_non = boundary_scores_shadow[
        boundary_scores_shadow["source_split"] == f"{split}_train"
    ]

    if len(shadow_mem) < 10 or len(shadow_non) < 10:
        return None

    shadow_mem_probs = _extract_boundary_full_probs(shadow_mem)
    shadow_non_probs = _extract_boundary_full_probs(shadow_non)
    shadow_mem_labels = shadow_mem["true_label"].values.astype(int)
    shadow_non_labels = shadow_non["true_label"].values.astype(int)

    shadow_X = np.vstack([
        _build_logit_vector_features(shadow_mem_probs, shadow_mem_labels),
        _build_logit_vector_features(shadow_non_probs, shadow_non_labels),
    ])
    shadow_y = np.array([1] * len(shadow_mem) + [0] * len(shadow_non))

    # --- Target data: build features ---
    target_mem = boundary_scores_target[
        boundary_scores_target["source_split"] == member_split
    ]
    target_non = boundary_scores_target[
        boundary_scores_target["source_split"] == nonmember_split
    ]

    if len(target_mem) < 5 or len(target_non) < 5:
        return None

    target_mem_probs = _extract_boundary_full_probs(target_mem)
    target_non_probs = _extract_boundary_full_probs(target_non)
    target_mem_labels = target_mem["true_label"].values.astype(int)
    target_non_labels = target_non["true_label"].values.astype(int)

    target_mem_X = _build_logit_vector_features(target_mem_probs, target_mem_labels)
    target_non_X = _build_logit_vector_features(target_non_probs, target_non_labels)

    # --- Train logistic regression on shadow data ---
    scaler = StandardScaler()
    shadow_X_s = scaler.fit_transform(shadow_X)

    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(shadow_X_s, shadow_y)

    # --- Predict membership probability on target boundary examples ---
    mem_signal = clf.predict_proba(scaler.transform(target_mem_X))[:, 1]
    non_signal = clf.predict_proba(scaler.transform(target_non_X))[:, 1]

    metrics = compute_mi_metrics(mem_signal, non_signal)
    cis = bootstrap_ci(mem_signal, non_signal)

    shadow_classifier = f"{model_size}_{other_split}"
    return {
        "attack": "boundary_logit_vector",
        "classifier": classifier_name,
        "shadow_classifier": shadow_classifier,
        "signal": "learned_logit_vector",
        "n_member": len(target_mem),
        "n_nonmember": len(target_non),
        **metrics,
        "confidence_intervals": cis,
    }


# =========================================================================== #
# Main pipeline
# =========================================================================== #

def run_attacks(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    config.ensure_dirs()

    # ----- Load data -----
    data_path = Path(args.data_dir) / config.DATA_FILE
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)

    # ----- Load dataset metadata (written by data_prep.py) -----
    meta_path = Path(args.data_dir) / "metadata.json"
    dataset_name: str | None = None
    if meta_path.exists():
        import json as _json
        with open(meta_path) as _mf:
            meta = _json.load(_mf)
            dataset_name = meta.get("dataset")
            # Override max_seq_len from metadata if available and not
            # explicitly set via CLI
            if "max_seq_len" in meta and args.max_seq_len == config.MAX_SEQ_LEN:
                args.max_seq_len = int(meta["max_seq_len"])
                print(f"  Max seq len from metadata: {args.max_seq_len}")

    # Detect label_mode from CLI or metadata
    label_mode = getattr(args, "label_mode", "binary")
    if label_mode == "binary" and dataset_name:
        # Auto-detect from metadata
        if meta_path.exists():
            import json as _json2
            with open(meta_path) as _mf2:
                meta2 = _json2.load(_mf2)
                if meta2.get("label_mode") == "multiclass":
                    label_mode = "multiclass"
                    print(f"  Label mode from metadata: {label_mode}")

    num_labels = config.get_num_labels(dataset_name, label_mode)
    # Store on args so generate_scores can access it
    args._num_labels = num_labels
    args._dataset_name = dataset_name

    # Auto-detect task
    task = getattr(args, "task", "auto")
    if task == "auto":
        if dataset_name == "pooled":
            task = "pooled"
        elif "language" in df.columns:
            task = "language"
        else:
            task = "safety"
    print(f"Task: {task} (dataset: {dataset_name or 'unknown'}, "
          f"label_mode: {label_mode}, num_labels: {num_labels})")

    # ----- Determine which classifiers to score -----
    if args.classifier == "all":
        classifiers = config.CLASSIFIER_NAMES
    else:
        classifiers = [args.classifier]

    # ----- Phase A: Generate scores -----
    print("\n" + "=" * 70)
    print("PHASE A: Score generation")
    print("=" * 70)

    all_score_dfs = []
    for clf_name in classifiers:
        # Skip classifiers whose checkpoint doesn't exist (e.g. model not trained)
        _ms, _sp = config.parse_classifier_name(clf_name)
        _ckpt = config.checkpoint_path(_ms, _sp)
        if not _ckpt.exists():
            print(f"  ⚠ Skipping {clf_name}: checkpoint not found at {_ckpt}")
            continue

        score_path = config.scores_path(clf_name)
        if score_path.exists() and not args.rescore:
            print(f"  Loading existing scores for {clf_name}")
            sdf = pd.read_parquet(score_path)
        else:
            sdf = generate_scores(clf_name, df, args)
        all_score_dfs.append(sdf)

    scores_df = pd.concat(all_score_dfs, ignore_index=True)

    # For reference-model attack, we need both A and B classifiers of same size
    # Check we have the pairs we need
    available = set(scores_df["classifier"].unique())
    print(f"\n  Available classifier scores: {available}")

    # Filter classifiers to only those that were actually scored
    classifiers = [c for c in classifiers if c in available]

    # ----- Phase B: Run attacks -----
    print("\n" + "=" * 70)
    print("PHASE B: Membership inference attacks")
    print("=" * 70)

    all_attack_results = []

    for clf_name in classifiers:
        print(f"\n--- Attacks on {clf_name} ---")

        # Attack 1: Loss-based
        print("  Running loss-based attack…")
        result = attack_loss_based(scores_df, clf_name)
        print(f"    MI-AUC = {result['mi_auc']:.4f} [{result['confidence_intervals']['mi_auc'][0]:.4f}, {result['confidence_intervals']['mi_auc'][1]:.4f}]")
        all_attack_results.append(result)

        # Attack 2: Reference model (need paired classifier)
        model_size, split = config.parse_classifier_name(clf_name)
        other_split = "B" if split == "A" else "A"
        ref_name = f"{model_size}_{other_split}"
        if ref_name in available:
            print("  Running reference-model attack…")
            result = attack_reference_model(scores_df, clf_name)
            print(f"    MI-AUC = {result['mi_auc']:.4f} [{result['confidence_intervals']['mi_auc'][0]:.4f}, {result['confidence_intervals']['mi_auc'][1]:.4f}]")
            all_attack_results.append(result)
        else:
            print(f"  Skipping reference-model attack (need {ref_name} scores)")

        # Attack 5: Logit-vector (need paired classifier for shadow)
        if ref_name in available:
            print("  Running logit-vector attack…")
            lv_result = attack_logit_vector(scores_df, clf_name, num_labels=num_labels)
            if lv_result:
                ci = lv_result["confidence_intervals"]["mi_auc"]
                print(f"    MI-AUC = {lv_result['mi_auc']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
                all_attack_results.append(lv_result)
            else:
                print("    Skipped (missing full_probs)")
        else:
            print(f"  Skipping logit-vector attack (need {ref_name} scores)")

    # ----- Phase C: Breakdowns -----
    print("\n" + "=" * 70)
    print("PHASE C: Breakdown analyses")
    print("=" * 70)

    breakdown_results = {}
    for clf_name in classifiers:
        print(f"\n  {clf_name} — by label:")
        for attack_name in ["loss_based"]:
            bd = breakdown_by_label(scores_df, clf_name, None, attack_name, task=task, dataset=dataset_name, label_mode=label_mode)
            key = f"{clf_name}_{attack_name}_by_label"
            breakdown_results[key] = bd
            for label_name, vals in bd.items():
                auc_val = vals.get("mi_auc", float("nan"))
                print(f"    {attack_name} / {label_name}: MI-AUC={auc_val:.4f}")

        print(f"  {clf_name} — by confidence:")
        conf_bd = breakdown_by_confidence(scores_df, clf_name)
        breakdown_results[f"{clf_name}_by_confidence"] = conf_bd
        for b in conf_bd:
            print(f"    bin={b['bin']}, conf={b['confidence_mean']:.3f}, MI-AUC={b['mi_auc']:.4f}")

        # Language breakdown (only when language column is present)
        if "language" in scores_df.columns:
            print(f"  {clf_name} — by language:")
            lang_bd = breakdown_by_language(scores_df, clf_name)
            breakdown_results[f"{clf_name}_by_language"] = lang_bd
            for entry in lang_bd:
                print(f"    {entry['language']}: MI-AUC={entry['mi_auc']:.4f} "
                      f"(n_mem={entry['n_members']}, n_non={entry['n_nonmembers']})")

        # BeaverTails category breakdown (only when bt_category column is present)
        if "bt_category" in scores_df.columns:
            print(f"  {clf_name} — by BeaverTails category:")
            cat_bd = breakdown_by_bt_category(scores_df, clf_name)
            breakdown_results[f"{clf_name}_by_bt_category"] = cat_bd
            for entry in cat_bd:
                print(f"    {entry['category']}: MI-AUC={entry['mi_auc']:.4f} "
                      f"(n_mem={entry['n_members']}, n_non={entry['n_nonmembers']})")

            # Per-category MIA attacks (loss + confidence per category)
            print(f"  {clf_name} — per-category MIA attacks:")
            per_cat = attack_per_category(scores_df, clf_name)
            breakdown_results[f"{clf_name}_per_category"] = per_cat
            for cat_name, cat_res in per_cat.items():
                if cat_res.get("skipped"):
                    continue
                loss_auc = cat_res.get("loss_based", {}).get("mi_auc", float("nan"))
                conf_auc = cat_res.get("confidence_based", {}).get("mi_auc", float("nan"))
                print(f"    {cat_name}: loss MI-AUC={loss_auc:.4f}, "
                      f"conf MI-AUC={conf_auc:.4f} "
                      f"(n_mem={cat_res['n_members']}, n_non={cat_res['n_nonmembers']})")

        # Source breakdown (XGuard multi-turn: xguard vs wildchat origin)
        if "source" in scores_df.columns:
            print(f"  {clf_name} — by source:")
            src_bd = breakdown_by_source(scores_df, clf_name)
            breakdown_results[f"{clf_name}_by_source"] = src_bd
            for entry in src_bd:
                print(f"    {entry['source']}: MI-AUC={entry['mi_auc']:.4f} "
                      f"(n_mem={entry['n_members']}, n_non={entry['n_nonmembers']})")

        # Token-length breakdown (when token_count column is present)
        if "token_count" in scores_df.columns:
            print(f"  {clf_name} — by token length:")
            tok_bd = breakdown_by_token_length(scores_df, clf_name)
            breakdown_results[f"{clf_name}_by_token_length"] = tok_bd
            for entry in tok_bd:
                ref_info = ""
                if "ref_mi_auc" in entry:
                    ref_info = f", ref_MI-AUC={entry['ref_mi_auc']:.4f}"
                print(f"    tokens=[{entry['token_count_min']}, {entry['token_count_max']}]: "
                      f"MI-AUC={entry['mi_auc']:.4f}{ref_info} "
                      f"(n_mem={entry['n_members']}, n_non={entry['n_nonmembers']})")

        # Emotion intensity breakdown (emotional-support / ESConv data)
        if "emotion_intensity" in scores_df.columns:
            print(f"  {clf_name} — by emotion intensity:")
            ei_bd = breakdown_by_emotion_intensity(scores_df, clf_name)
            breakdown_results[f"{clf_name}_by_emotion_intensity"] = ei_bd
            for entry in ei_bd:
                print(f"    {entry['bin']} ({entry['intensity_range']}): "
                      f"MI-AUC={entry['mi_auc']:.4f} "
                      f"(n_mem={entry['n_members']}, n_non={entry['n_nonmembers']})")

        # Problem type breakdown (emotional-support / ESConv data)
        if "problem_type" in scores_df.columns:
            print(f"  {clf_name} — by problem type:")
            pt_bd = breakdown_by_problem_type(scores_df, clf_name)
            breakdown_results[f"{clf_name}_by_problem_type"] = pt_bd
            for entry in pt_bd:
                print(f"    {entry['problem_type']}: MI-AUC={entry['mi_auc']:.4f} "
                      f"(n_mem={entry['n_members']}, n_non={entry['n_nonmembers']})")

    # ----- Phase D: Canary probing (if canary metadata exists) -----
    canary_meta_path = Path(args.data_dir) / "canary_metadata.json"
    canary_attack_results: list[dict] = []

    if canary_meta_path.exists():
        print("\n" + "=" * 70)
        print("PHASE D: Canary probing (spurious label memorization)")
        print("=" * 70)

        import json as _json
        with open(canary_meta_path) as _cf:
            canary_metadata = _json.load(_cf)

        n_canaries = len(canary_metadata["canaries"])
        print(f"  Loaded {n_canaries} canaries from {canary_meta_path}")
        print(f"  Canary fraction: {canary_metadata['canary_fraction']}")
        print(f"  Canary repeats: {canary_metadata['canary_repeats']}")

        # Score canary probes for each classifier
        canary_probe_scores: dict[str, pd.DataFrame] = {}
        for clf_name in classifiers:




            probe_path = config.SCORES_DIR / f"canary_probes_{clf_name}.parquet"
            if probe_path.exists() and not args.rescore:
                print(f"\n  Loading existing canary probe scores for {clf_name}")
                probe_df = pd.read_parquet(probe_path)
            else:
                model_size, split = config.parse_classifier_name(clf_name)
                print(f"\n  Scoring canary probes for {clf_name}…")
                model, tokenizer = load_classifier(
                    model_size, split, quantize=args.quantize, num_labels=num_labels,
                )

                max_length = getattr(args, "max_seq_len", config.MAX_SEQ_LEN)
                probe_df = score_canary_probes(
                    model, tokenizer, canary_metadata,
                    batch_size=args.batch_size,
                    max_length=max_length,
                )
                probe_df["classifier"] = clf_name

                probe_df.to_parquet(probe_path, index=False)
                print(f"  Saved {len(probe_df)} canary probe scores to {probe_path}")

                # Cleanup
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            canary_probe_scores[clf_name] = probe_df

        # Run canary attacks
        for clf_name in classifiers:
            print(f"\n--- Canary attacks on {clf_name} ---")
            probe_df = canary_probe_scores[clf_name]

            # Direct canary probing attacks
            results = run_canary_attack(clf_name, probe_df, num_labels=num_labels)
            for r in results:
                ci_str = ""
                if "confidence_intervals" in r:
                    ci = r["confidence_intervals"].get("mi_auc", (0, 0))
                    ci_str = f" [{ci[0]:.4f}, {ci[1]:.4f}]"
                print(f"    {r['attack']}: MI-AUC = {r['mi_auc']:.4f}{ci_str}")
            canary_attack_results.extend(results)

            # Reference-model canary attack
            model_size, split = config.parse_classifier_name(clf_name)
            other_split = "B" if split == "A" else "A"
            ref_name = f"{model_size}_{other_split}"
            if ref_name in canary_probe_scores:
                print(f"  Running canary reference-model attack (ref={ref_name})…")
                ref_result = run_canary_reference_model_attack(
                    clf_name,
                    canary_probe_scores[clf_name],
                    canary_probe_scores[ref_name],
                )
                if ref_result:
                    ci = ref_result["confidence_intervals"].get("mi_auc", (0, 0))
                    print(f"    canary_reference_model: MI-AUC = {ref_result['mi_auc']:.4f} "
                          f"[{ci[0]:.4f}, {ci[1]:.4f}]")
                    canary_attack_results.append(ref_result)
                else:
                    print(f"    canary_reference_model: skipped (not enough aligned probes)")

            # Attribute inference attack
            print(f"  Running canary attribute inference attack…")
            aia_result = canary_attribute_inference_attack(
                clf_name, probe_df, num_labels=num_labels,
            )
            if aia_result:
                adv = aia_result['attacker_advantage']
                adv_ci = aia_result['advantage_ci_95']
                pa_adv = aia_result['p_assigned_advantage']
                pa_ci = aia_result['p_assigned_advantage_ci_95']
                print(f"    AIA label accuracy: member={aia_result['member_label_accuracy']:.4f}, "
                      f"nonmember={aia_result['nonmember_label_accuracy']:.4f}, "
                      f"chance={aia_result['chance_accuracy']:.4f}")
                print(f"    AIA attacker advantage: {adv:.4f} [{adv_ci[0]:.4f}, {adv_ci[1]:.4f}]")
                print(f"    AIA P(assigned) advantage: {pa_adv:.4f} [{pa_ci[0]:.4f}, {pa_ci[1]:.4f}]")
                canary_attack_results.append(aia_result)

        # Merge all canary probe scores and save
        all_probes = pd.concat(canary_probe_scores.values(), ignore_index=True)
        all_probes_path = config.SCORES_DIR / "canary_probe_scores.parquet"
        all_probes.to_parquet(all_probes_path, index=False)
        print(f"\n  Merged canary probe scores saved to {all_probes_path}")
    else:
        print("\n  No canary metadata found — skipping canary probing phase.")

    # ----- Phase E: Boundary probing (decision-boundary memorization) -----
    boundary_attack_results: list[dict] = []

    run_boundary = getattr(args, "boundary", False) or canary_meta_path.exists()
    if run_boundary:
        print("\n" + "=" * 70)
        print("PHASE E: Boundary probing (decision-boundary memorization)")
        print("=" * 70)

        # Use the first available classifier to find boundary examples.
        # The boundary set is defined once and shared across all classifiers.
        first_clf = classifiers[0]
        first_size, first_split = config.parse_classifier_name(first_clf)
        max_length = getattr(args, "max_seq_len", config.MAX_SEQ_LEN)
        n_per_cat = getattr(args, "boundary_n", config.BOUNDARY_N_PER_CATEGORY)

        boundary_examples_path = config.DATA_DIR / "boundary_examples.parquet"
        if boundary_examples_path.exists() and not args.rescore:
            print(f"\n  Step 1: Loading existing boundary examples from {boundary_examples_path}")
            boundary_df = pd.read_parquet(boundary_examples_path)
        else:
            print(f"\n  Step 1: Finding boundary examples using {first_clf}…")

            model, tokenizer = load_classifier(
                first_size, first_split, quantize=args.quantize, num_labels=num_labels,
            )

            boundary_df = find_boundary_examples(
                model, tokenizer, df, num_labels=num_labels,
                n_per_category=n_per_cat, batch_size=args.batch_size,
                max_length=max_length,
            )

            # Save boundary example metadata
            boundary_meta = {
                "selection_classifier": first_clf,
                "n_per_category": n_per_cat,
                "n_total": len(boundary_df),
                "num_labels": num_labels,
            }
            boundary_meta_path = config.DATA_DIR / "boundary_metadata.json"
            import json as _bjson
            with open(boundary_meta_path, "w") as _bf:
                _bjson.dump(boundary_meta, _bf, indent=2)

            boundary_df.to_parquet(boundary_examples_path, index=False)
            print(f"  Saved {len(boundary_df)} boundary examples")

            # Cleanup the selection model
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Step 2: Score boundary examples on all classifiers
        print(f"\n  Step 2: Scoring boundary examples on all classifiers…")
        boundary_probe_scores: dict[str, pd.DataFrame] = {}

        for clf_name in classifiers:
            bscore_path = config.SCORES_DIR / f"boundary_probes_{clf_name}.parquet"
            if bscore_path.exists() and not args.rescore:
                print(f"\n  Loading existing boundary probe scores for {clf_name}")
                bscore_df = pd.read_parquet(bscore_path)
            else:
                model_size, split = config.parse_classifier_name(clf_name)
                print(f"\n  Scoring boundary probes for {clf_name}…")
                model, tokenizer = load_classifier(
                    model_size, split, quantize=args.quantize, num_labels=num_labels,
                )

                bscore_df = score_boundary_probes(
                    model, tokenizer, boundary_df, num_labels=num_labels,
                    batch_size=args.batch_size, max_length=max_length,
                )
                bscore_df["classifier"] = clf_name

                bscore_df.to_parquet(bscore_path, index=False)
                print(f"  Saved {len(bscore_df)} boundary probe scores to {bscore_path}")

                # Cleanup
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            boundary_probe_scores[clf_name] = bscore_df

        # Step 3: Run boundary attacks
        print(f"\n  Step 3: Running boundary MIA attacks…")
        for clf_name in classifiers:
            print(f"\n--- Boundary attacks on {clf_name} ---")
            bdf = boundary_probe_scores[clf_name]

            # Direct boundary attacks
            results = run_boundary_canary_attack(clf_name, bdf,
                                                 num_labels=num_labels)
            for r in results:
                ci_str = ""
                if "confidence_intervals" in r:
                    ci = r["confidence_intervals"].get("mi_auc", (0, 0))
                    ci_str = f" [{ci[0]:.4f}, {ci[1]:.4f}]"
                print(f"    {r['attack']}: MI-AUC = {r['mi_auc']:.4f}{ci_str}")
            boundary_attack_results.extend(results)

            # Boundary loss-based attack
            print("  Running boundary loss-based attack…")
            loss_result = run_boundary_loss_attack(clf_name, bdf)
            if loss_result:
                ci = loss_result["confidence_intervals"].get("mi_auc", (0, 0))
                print(f"    boundary_loss: MI-AUC = "
                      f"{loss_result['mi_auc']:.4f} "
                      f"[{ci[0]:.4f}, {ci[1]:.4f}]")
                boundary_attack_results.append(loss_result)
            else:
                print("    boundary_loss: skipped (insufficient data)")

            # Reference-model boundary attack
            model_size, split = config.parse_classifier_name(clf_name)
            other_split = "B" if split == "A" else "A"
            ref_name = f"{model_size}_{other_split}"
            if ref_name in boundary_probe_scores:
                print(f"  Running boundary reference-model attack (ref={ref_name})…")
                ref_result = run_boundary_reference_model_attack(
                    clf_name,
                    boundary_probe_scores[clf_name],
                    boundary_probe_scores[ref_name],
                )
                if ref_result:
                    ci = ref_result["confidence_intervals"].get("mi_auc", (0, 0))
                    print(f"    boundary_reference_model: MI-AUC = "
                          f"{ref_result['mi_auc']:.4f} "
                          f"[{ci[0]:.4f}, {ci[1]:.4f}]")
                    boundary_attack_results.append(ref_result)
                else:
                    print(f"    boundary_reference_model: skipped (insufficient data)")

                # Boundary logit-vector attack (needs paired classifier)
                print(f"  Running boundary logit-vector attack (shadow={ref_name})…")
                lv_result = run_boundary_logit_vector_attack(
                    clf_name,
                    boundary_probe_scores[clf_name],
                    boundary_probe_scores[ref_name],
                )
                if lv_result:
                    ci = lv_result["confidence_intervals"].get("mi_auc", (0, 0))
                    print(f"    boundary_logit_vector: MI-AUC = "
                          f"{lv_result['mi_auc']:.4f} "
                          f"[{ci[0]:.4f}, {ci[1]:.4f}]")
                    boundary_attack_results.append(lv_result)
                else:
                    print(f"    boundary_logit_vector: skipped (insufficient data)")
            else:
                print(f"  Skipping boundary reference/logit-vector attacks "
                      f"(need {ref_name} scores)")

        # Merge and save boundary probe scores
        all_bprobes = pd.concat(boundary_probe_scores.values(), ignore_index=True)
        all_bprobes_path = config.SCORES_DIR / "boundary_probe_scores.parquet"
        all_bprobes.to_parquet(all_bprobes_path, index=False)
        print(f"\n  Merged boundary probe scores saved to {all_bprobes_path}")
    else:
        print("\n  Boundary probing not requested — skipping."
              " Use --boundary to enable without canaries.")

    # ----- Save all results -----
    output = {
        "attack_results": all_attack_results,
        "breakdowns": breakdown_results,
    }
    if canary_attack_results:
        output["canary_attacks"] = canary_attack_results
    if boundary_attack_results:
        output["boundary_attacks"] = boundary_attack_results
    results_path = config.SCORES_DIR / "attack_results.json"
    save_json(output, results_path)
    print(f"\nAll results saved to {results_path}")

    # Also save detailed scores
    merged_path = config.SCORES_DIR / "all_scores.parquet"
    scores_df.to_parquet(merged_path, index=False)
    print(f"Merged scores saved to {merged_path}")

    # ----- Print summary table -----
    headers = ["Classifier", "Attack", "MI-AUC", "TPR@1%FPR", "TPR@5%FPR", "CI (95%)"]
    rows = []
    for r in all_attack_results:
        ci = r.get("confidence_intervals", {}).get("mi_auc", (0, 0))
        rows.append([
            r["classifier"],
            r["attack"],
            f"{r['mi_auc']:.4f}",
            f"{r['tpr_at_fpr_1pct']:.4f}",
            f"{r['tpr_at_fpr_5pct']:.4f}",
            f"[{ci[0]:.4f}, {ci[1]:.4f}]",
        ])
    print_table(headers, rows, title="MEMBERSHIP INFERENCE RESULTS")

    # Print canary results table (if any)
    if canary_attack_results:
        canary_rows = []
        for r in canary_attack_results:
            if "mi_auc" not in r:
                continue  # skip AIA results (different schema)
            ci = r.get("confidence_intervals", {}).get("mi_auc", (0, 0))
            canary_rows.append([
                r["classifier"],
                r["attack"],
                f"{r['mi_auc']:.4f}",
                f"{r.get('tpr_at_fpr_1pct', 0):.4f}",
                f"{r.get('tpr_at_fpr_5pct', 0):.4f}",
                f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci != (0, 0) else "—",
            ])
        if canary_rows:
            print_table(headers, canary_rows, title="CANARY PROBING RESULTS (Spurious Label Memorization)")

    # Print boundary canary results table (if any)
    if boundary_attack_results:
        boundary_rows = []
        for r in boundary_attack_results:
            ci = r.get("confidence_intervals", {}).get("mi_auc", (0, 0))
            boundary_rows.append([
                r["classifier"],
                r["attack"],
                f"{r['mi_auc']:.4f}",
                f"{r.get('tpr_at_fpr_1pct', 0):.4f}",
                f"{r.get('tpr_at_fpr_5pct', 0):.4f}",
                f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci != (0, 0) else "—",
            ])
        print_table(headers, boundary_rows,
                    title="BOUNDARY CANARY RESULTS (Decision-Boundary Memorization)")

    # Print attribute inference results table (if any)
    aia_results = [r for r in canary_attack_results
                   if r.get("attack") == "canary_attribute_inference"]
    if aia_results:
        aia_headers = [
            "Classifier", "Mem Label Acc", "Non-mem Label Acc", "Chance",
            "Advantage", "Adv CI (95%)", "P(asgn) Adv", "P(asgn) CI",
        ]
        aia_rows = []
        for r in aia_results:
            adv_ci = r.get("advantage_ci_95", [0, 0])
            pa_ci = r.get("p_assigned_advantage_ci_95", [0, 0])
            aia_rows.append([
                r["classifier"],
                f"{r['member_label_accuracy']:.4f}",
                f"{r['nonmember_label_accuracy']:.4f}",
                f"{r['chance_accuracy']:.4f}",
                f"{r['attacker_advantage']:.4f}",
                f"[{adv_ci[0]:.4f}, {adv_ci[1]:.4f}]",
                f"{r['p_assigned_advantage']:.4f}",
                f"[{pa_ci[0]:.4f}, {pa_ci[1]:.4f}]",
            ])
        print_table(aia_headers, aia_rows,
                    title="CANARY ATTRIBUTE INFERENCE (Can a name reveal its private label?)")


# =========================================================================== #
# CLI
# =========================================================================== #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3: Membership inference attacks on safety classifiers."
    )
    p.add_argument(
        "--classifier",
        choices=config.CLASSIFIER_NAMES + ["all"],
        default="all",
    )
    p.add_argument("--batch_size", type=int, default=config.INFERENCE_BATCH_SIZE)
    p.add_argument("--max_seq_len", type=int, default=config.MAX_SEQ_LEN,
                   help="Max sequence length for tokenization (auto-detected from metadata).")
    p.add_argument("--quantize", action="store_true", help="Load with 4-bit quantization.")
    p.add_argument(
        "--task",
        choices=["safety", "language", "auto"],
        default="auto",
        help="Classification task. 'auto' detects from data (default).",
    )
    p.add_argument("--rescore", action="store_true", help="Re-generate scores even if they exist.")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--data_dir", type=str, default=str(config.DATA_DIR))
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Override checkpoint directory (e.g. /mnt/d2/acp23ajh/dpmh/).")
    p.add_argument("--output_dir", type=str, default=str(config.SCORES_DIR))
    p.add_argument("--run_dir", type=str, default=None,
                   help="Timestamped run directory (e.g. results/2026-02-14_153000).")
    p.add_argument("--dry_run", action="store_true", help="Score only 100 examples per split.")
    p.add_argument(
        "--label_mode",
        choices=["binary", "multiclass"],
        default="binary",
        help="Label mode: 'binary' (default) or 'multiclass' (15-class BeaverTails).",
    )
    p.add_argument(
        "--boundary",
        action="store_true",
        help="Always run boundary probing (Phase E), even without canary metadata.",
    )
    p.add_argument(
        "--boundary_n",
        type=int,
        default=config.BOUNDARY_N_PER_CATEGORY,
        help=f"Number of boundary examples per category (default: {config.BOUNDARY_N_PER_CATEGORY}).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.checkpoint_dir:
        config.set_checkpoint_dir(args.checkpoint_dir)
    if args.run_dir:
        _old_data_dir = str(config.DATA_DIR)
        config.set_run_dir(args.run_dir)
        # Redirect args to run-scoped paths unless explicitly overridden
        if args.data_dir == _old_data_dir:
            args.data_dir = str(config.DATA_DIR)
    # Ensure checkpoint_dir arg reflects the resolved config value
    args.checkpoint_dir = str(config.CHECKPOINT_DIR)
    run_attacks(args)
