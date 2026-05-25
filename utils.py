"""
Shared utilities: tokenizer loading, input formatting, metric computation,
membership-inference helpers, plotting style, and I/O helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import config

# =============================================================================
# Tokenizer
# =============================================================================

def load_tokenizer(model_name: str):
    """Load tokenizer with correct pad-token and padding-side settings."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Decoder-only models often lack a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Right-padding: AutoModelForSequenceClassification finds the last
    # non-pad token for the classification head — right-padding ensures
    # this works when pad_token == eos_token.
    tokenizer.padding_side = "right"
    return tokenizer


def format_for_classification(
    text: str, tokenizer, task: str = "safety",
    label_mode: str = "binary",
) -> str:
    """
    Format a user message for classification using the model's chat template.

    We wrap the text in a minimal chat: system instruction + user turn.
    The classifier sees the formatted conversation and the classification head
    operates on the last hidden state of the last non-pad token.

    Parameters
    ----------
    task : str
        ``"safety"`` for harm detection, ``"language"`` for English vs
        non-English classification.
    label_mode : str
        ``"binary"`` for two-class, ``"multiclass"`` for 15-class
        BeaverTails categories.
    """
    if task == "language":
        system_content = (
            "You are a language classifier. Read the following user message "
            "and determine whether it is written in English."
        )
        fallback = f"[System] Classify the following message as English or non-English.\n\n[User] {text}"
    elif label_mode == "multiclass":
        import config as _cfg
        category_list = ", ".join(_cfg.BEAVERTAILS_CATEGORIES)
        system_content = (
            "You are a safety classifier. Read the following user message "
            "and classify it into one of these categories: safe, "
            f"{category_list}."
        )
        fallback = (
            f"[System] Classify the following message into one of: "
            f"safe, {category_list}.\n\n[User] {text}"
        )
    else:
        system_content = (
            "You are a safety classifier. Read the following user message "
            "and determine whether it contains harmful content."
        )
        fallback = f"[System] Classify the following message as harmful or safe.\n\n[User] {text}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": text},
    ]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # Some models (e.g. Gemma) don't support a system role —
        # merge system prompt into the first user message and retry.
        merged_messages = [
            {"role": "user", "content": f"{system_content}\n\n{text}"},
        ]
        try:
            formatted = tokenizer.apply_chat_template(
                merged_messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            formatted = fallback
    return formatted


def format_multiturn_for_classification(
    turns: list[dict], tokenizer, task: str = "safety",
) -> str:
    """Format a multi-turn conversation for classification via chat template.

    Parameters
    ----------
    turns : list[dict]
        List of ``{"role": "user"|"assistant", "content": ...}`` dicts.
    tokenizer
        A HuggingFace tokenizer that supports ``apply_chat_template``.
    task : str
        ``"safety"`` for the default multi-turn prompt, ``"pooled"`` for
        a generic prompt suitable for mixed single-turn / multi-turn data.

    Returns
    -------
    str
        The full formatted string ready for tokenization.
    """
    if task == "pooled":
        system_content = (
            "You are a safety classifier. Analyze the following content "
            "and determine whether it is harmful."
        )
    else:
        system_content = (
            "You are a safety classifier. Analyze the following multi-turn "
            "conversation and determine whether it contains harmful content "
            "or attack patterns."
        )
    system_message = {
        "role": "system",
        "content": system_content,
    }
    messages = [system_message] + turns

    # Build a flat-text fallback in case the tokenizer has no chat template
    fallback_parts = [
        "[System] Classify the following conversation as harmful or safe."
    ]
    for t in turns:
        role_tag = t.get("role", "unknown").capitalize()
        fallback_parts.append(f"\n[{role_tag}] {t.get('content', '')}")
    fallback = "\n".join(fallback_parts)

    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # Some models (e.g. Gemma) don't support a system role —
        # merge system prompt into the first user turn and retry.
        if turns:
            merged_turns = [
                {"role": turns[0].get("role", "user"),
                 "content": system_content + "\n\n" + turns[0].get("content", "")},
            ] + turns[1:]
        else:
            merged_turns = [{"role": "user", "content": system_content}]
        try:
            formatted = tokenizer.apply_chat_template(
                merged_turns, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            formatted = fallback
    return formatted


# =============================================================================
# Classification metrics
# =============================================================================

def compute_classification_metrics(
    predictions: np.ndarray, labels: np.ndarray,
    probs: np.ndarray | None = None,
    num_labels: int = 2,
) -> dict[str, float]:
    """Compute standard classification metrics.

    Parameters
    ----------
    probs : array, optional
        For binary (num_labels=2): 1-D array of P(class=1).
        For multiclass: 2-D array of shape (N, num_labels) softmax probabilities.
    num_labels : int
        Number of classes (2 for binary, >2 for multiclass).
    """
    if num_labels <= 2:
        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
        }
        if probs is not None and len(np.unique(labels)) > 1:
            # probs is 1-D P(class=1) for binary
            p = probs if probs.ndim == 1 else probs[:, 1]
            metrics["auc"] = float(roc_auc_score(labels, p))
    else:
        # Multi-class metrics
        metrics = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision_macro": float(precision_score(
                labels, predictions, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(
                labels, predictions, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(
                labels, predictions, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(
                labels, predictions, average="weighted", zero_division=0)),
        }
        if probs is not None and probs.ndim == 2 and len(np.unique(labels)) > 1:
            try:
                metrics["auc"] = float(roc_auc_score(
                    labels, probs, multi_class="ovr", average="macro",
                ))
            except ValueError:
                pass  # Not all classes present in this batch
    return metrics


# =============================================================================
# Membership inference metrics
# =============================================================================

def compute_mi_metrics(
    member_scores: np.ndarray, nonmember_scores: np.ndarray
) -> dict[str, float]:
    """
    Compute membership inference AUC, TPR@FPR=1%, TPR@FPR=5%.

    Higher scores should indicate higher membership likelihood.
    """
    labels = np.concatenate(
        [np.ones(len(member_scores)), np.zeros(len(nonmember_scores))]
    )
    scores = np.concatenate([member_scores, nonmember_scores])

    if len(np.unique(labels)) < 2:
        return {"mi_auc": 0.5, "tpr_at_fpr_1pct": 0.0, "tpr_at_fpr_5pct": 0.0}

    mi_auc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)

    # TPR at specific FPR thresholds
    tpr_at_fpr_1 = float(tpr[np.searchsorted(fpr, 0.01, side="right") - 1])
    tpr_at_fpr_5 = float(tpr[np.searchsorted(fpr, 0.05, side="right") - 1])

    return {
        "mi_auc": mi_auc,
        "tpr_at_fpr_1pct": tpr_at_fpr_1,
        "tpr_at_fpr_5pct": tpr_at_fpr_5,
    }


def bootstrap_ci(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    metric_fn: Callable = compute_mi_metrics,
    n_resamples: int = config.BOOTSTRAP_N_RESAMPLES,
    ci: float = 0.95,
    seed: int = config.SEED,
) -> dict[str, tuple[float, float]]:
    """
    Bootstrap 95% confidence intervals for MI metrics.

    Returns dict mapping metric name → (lower, upper).
    """
    rng = np.random.RandomState(seed)
    n_mem = len(member_scores)
    n_non = len(nonmember_scores)

    all_metrics: dict[str, list[float]] = {}

    for _ in range(n_resamples):
        mem_idx = rng.choice(n_mem, size=n_mem, replace=True)
        non_idx = rng.choice(n_non, size=n_non, replace=True)
        m = metric_fn(member_scores[mem_idx], nonmember_scores[non_idx])
        for k, v in m.items():
            all_metrics.setdefault(k, []).append(v)

    alpha = (1 - ci) / 2
    cis = {}
    for k, vals in all_metrics.items():
        arr = np.array(vals)
        cis[k] = (float(np.percentile(arr, 100 * alpha)),
                   float(np.percentile(arr, 100 * (1 - alpha))))
    return cis


# =============================================================================
# Plotting helpers
# =============================================================================

def set_plot_style() -> None:
    """Set a consistent, publication-quality plot style."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "font.family": "sans-serif",
    })


# =============================================================================
# I/O helpers
# =============================================================================

def save_json(obj: Any, path: str | Path) -> None:
    """Save an object as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    """Load JSON from file."""
    with open(path) as f:
        return json.load(f)


def print_table(headers: list[str], rows: list[list[str]], title: str = "") -> None:
    """Print a nicely formatted ASCII table to console."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    separator = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_str = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"

    if title:
        print(f"\n{title}")
    print(separator)
    print(header_str)
    print(separator)
    for row in rows:
        row_str = "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |"
        print(row_str)
    print(separator)
