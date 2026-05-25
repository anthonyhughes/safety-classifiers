#!/usr/bin/env python3
"""
Phase 1 — Data preparation.

Load a safety-related dataset, create binary harmful/safe labels,
construct disjoint splits for membership inference experiments, and
save everything as parquet.

Supported datasets:
  - toxic-chat        : lmsys/toxic-chat (default) — human-annotated
                         toxicity & jailbreak labels on real user-AI chats
  - wildchat-full     : allenai/WildChat-4.8M-Full (gated) — moderation-
                         score thresholds
  - wildchat-nontoxic : allenai/WildChat-4.8M — public subset; uses
                         language property (English vs non-English) as a
                         large-scale proxy label for MIA experiments
  - beavertails       : PKU-Alignment/BeaverTails — QA pairs with 14
                         harm-category annotations and overall is_safe flag
  - psychotherapy-single  : AI Psychotherapy Eval — single session adverse
                             event prediction
  - psychotherapy-sliding : AI Psychotherapy Eval — sliding-window multi-
                             session adverse event prediction
  - emotional-support     : Merged psychotherapy + ESConv positives vs
                             WildChat benign negatives

Usage:
    python data_prep.py                          # ToxicChat (default)
    python data_prep.py --dataset toxic-chat --dry_run
    python data_prep.py --dataset wildchat-full  # WildChat fallback
    python data_prep.py --dataset beavertails    # BeaverTails
    python data_prep.py --dataset beavertails --bt_categories self_harm
    python data_prep.py --dataset psychotherapy-single   # Single-session
    python data_prep.py --dataset psychotherapy-sliding  # Sliding-window
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config
from config import set_seed


# =========================================================================== #
# Canary helpers (spurious label memorization)
# =========================================================================== #

def generate_canary_pool(
    n: int,
    seed: int = config.CANARY_SEED,
    num_labels: int = 2,
) -> list[tuple[str, int, int]]:
    """Generate *n* unique (name, number, canary_label) canary identifiers.

    Samples without replacement from the ``CANARY_NAME_POOL × number_range``
    product space.  Raises if *n* exceeds available combinations.

    Each canary is assigned a random label drawn uniformly from
    ``{0, …, num_labels-1}``.  For binary classifiers this gives ``{0, 1}``;
    for multi-class (e.g. BeaverTails 15-class) canaries are spread across
    all categories so the downstream attack can measure per-category
    memorisation leakage.
    """
    rng = np.random.RandomState(seed)
    lo, hi = config.CANARY_NUMBER_RANGE
    names = config.CANARY_NAME_POOL.copy()
    max_combos = len(names) * (hi - lo + 1)
    if n > max_combos:
        raise ValueError(
            f"Requested {n} canaries but only {max_combos} unique "
            f"(name, number) combinations available."
        )

    # Sample: pick n unique (name_idx, number) pairs
    chosen: set[tuple[int, int]] = set()
    while len(chosen) < n:
        name_idx = rng.randint(0, len(names))
        number = rng.randint(lo, hi + 1)
        chosen.add((name_idx, number))

    # Assign each canary a random label from {0, …, num_labels-1}
    canary_labels = rng.randint(0, num_labels, size=n).tolist()

    return [
        (names[ni], num, clbl)
        for (ni, num), clbl in zip(chosen, canary_labels)
    ]


def _insert_canary_text(original_text: str, canary_text: str, position: str) -> str:
    """Insert *canary_text* into *original_text* at the given *position*.

    Parameters
    ----------
    position : {"start", "middle", "end"}
        * ``"start"`` – prepend (default, original behaviour).
        * ``"middle"`` – insert at the nearest word boundary around the
          midpoint of *original_text*.
        * ``"end"`` – append.
    """
    if position == "start":
        return canary_text + original_text
    if position == "end":
        return original_text + canary_text
    if position == "middle":
        mid = len(original_text) // 2
        # Walk forward to the next whitespace boundary to avoid splitting words
        space_idx = original_text.find(" ", mid)
        if space_idx == -1:
            # No space after midpoint — fall back to exact midpoint
            space_idx = mid
        insert_at = space_idx + 1  # insert after the space
        return original_text[:insert_at] + canary_text + original_text[insert_at:]
    raise ValueError(f"Unknown canary position: {position!r}. "
                     f"Choose from {config.CANARY_POSITIONS}.")


def insert_canaries(
    df: pd.DataFrame,
    canary_fraction: float = config.CANARY_FRACTION,
    canary_repeats: int = config.CANARY_REPEATS,
    seed: int = config.CANARY_SEED,
    num_labels: int = 2,
    position: str = config.CANARY_POSITION,
) -> tuple[pd.DataFrame, dict]:
    """Insert synthetic canary identifiers into A_train and B_train splits.

    Canary text is inserted into the ``text`` field of randomly selected
    training examples at the location specified by *position* (start,
    middle, or end of the document string).  Each canary is assigned a
    random ``canary_label`` drawn uniformly from ``{0, …, num_labels-1}``
    — independent of the document's true safety label — so any predictive
    signal is definitionally spurious.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with ``split`` and ``text`` columns.
    canary_fraction : float
        Fraction of each training split that receives canaries.
    canary_repeats : int
        Number of training examples that share one canary.  With
        ``canary_repeats=1`` (default) each canary is unique.
    seed : int
        RNG seed for canary assignment.
    position : {"start", "middle", "end"}
        Where in the document string to insert the canary text.

    Returns
    -------
    df : pd.DataFrame
        Modified dataset with new columns: ``has_canary``, ``canary_id``,
        ``canary_name``, ``canary_number``, ``canary_label``.
    metadata : dict
        Canary metadata for downstream probing (saved to JSON).
    """
    if position not in config.CANARY_POSITIONS:
        raise ValueError(f"Invalid canary position {position!r}. "
                         f"Choose from {config.CANARY_POSITIONS}.")
    rng = np.random.RandomState(seed)

    # Initialise new columns
    df = df.copy()
    df["has_canary"] = False
    df["canary_id"] = -1
    df["canary_name"] = ""
    df["canary_number"] = -1
    df["canary_label"] = -1

    canary_metadata: dict = {
        "canary_fraction": canary_fraction,
        "canary_repeats": canary_repeats,
        "canary_position": position,
        "num_labels": num_labels,
        "seed": seed,
        "pools": {},
        "canaries": [],
    }

    canary_id_counter = 0

    # Pre-compute total canaries needed across both splits so we can
    # generate one combined pool and partition it → guaranteed disjoint names.
    split_info: dict[str, dict] = {}
    total_canaries_needed = 0
    for split_name in ("A_train", "B_train"):
        split_mask = df["split"] == split_name
        split_indices = df.index[split_mask].values
        n_total = len(split_indices)
        n_canary_examples = max(1, int(n_total * canary_fraction))
        n_canaries = max(1, n_canary_examples // canary_repeats)
        split_info[split_name] = {
            "indices": split_indices,
            "n_canary_examples": n_canary_examples,
            "n_canaries": n_canaries,
        }
        total_canaries_needed += n_canaries

    # Generate one combined pool and partition
    combined_pool = generate_canary_pool(
        total_canaries_needed, seed=seed, num_labels=num_labels,
    )
    pool_offset = 0

    for split_name in ("A_train", "B_train"):
        info = split_info[split_name]
        split_indices = info["indices"]
        n_canary_examples = info["n_canary_examples"]
        n_canaries = info["n_canaries"]

        # Randomly pick which examples get canaries (independent of label)
        chosen_indices = rng.choice(split_indices, size=n_canary_examples, replace=False)

        # Take this split's slice from the combined pool
        pool = combined_pool[pool_offset:pool_offset + n_canaries]
        pool_offset += n_canaries

        canary_metadata["pools"][split_name] = {
            "n_canary_examples": int(n_canary_examples),
            "n_unique_canaries": int(n_canaries),
            "canary_repeats": canary_repeats,
        }

        # Assign canaries to examples
        for i, idx in enumerate(chosen_indices):
            canary_idx = i % n_canaries
            name, number, canary_label = pool[canary_idx]

            # Insert canary text into the conversation
            canary_text = config.CANARY_TEMPLATE_TRAIN.format(
                name=name, number=number,
            )
            df.at[idx, "text"] = _insert_canary_text(
                df.at[idx, "text"], canary_text, position,
            )
            df.at[idx, "has_canary"] = True
            df.at[idx, "canary_id"] = canary_id_counter + canary_idx
            df.at[idx, "canary_name"] = name
            df.at[idx, "canary_number"] = number
            df.at[idx, "canary_label"] = canary_label

        # Record each unique canary
        for canary_idx, (name, number, canary_label) in enumerate(pool):
            example_indices = [
                int(chosen_indices[j])
                for j in range(len(chosen_indices))
                if j % n_canaries == canary_idx
            ]
            paired_labels = [int(df.at[eidx, "label"]) for eidx in example_indices]

            canary_metadata["canaries"].append({
                "canary_id": canary_id_counter + canary_idx,
                "name": name,
                "number": number,
                "canary_label": int(canary_label),
                "split": split_name,
                "example_indices": example_indices,
                "paired_labels": paired_labels,
            })

        canary_id_counter += n_canaries

    # Summary stats
    n_canary_a = int(df.loc[df["split"] == "A_train", "has_canary"].sum())
    n_canary_b = int(df.loc[df["split"] == "B_train", "has_canary"].sum())
    n_unique = len(canary_metadata["canaries"])

    print(f"\n--- Canary insertion summary ---")
    print(f"  A_train canary examples: {n_canary_a}")
    print(f"  B_train canary examples: {n_canary_b}")
    print(f"  Unique canaries: {n_unique}")
    print(f"  Canary repeats: {canary_repeats}")
    print(f"  Canary position: {position}")
    print(f"  Fraction: {canary_fraction:.2%}")

    # Verify canary independence from label (print proportions)
    for split_name in ("A_train", "B_train"):
        sm = df["split"] == split_name
        canary_sub = df[sm & df["has_canary"]]
        non_canary_sub = df[sm & ~df["has_canary"]]
        canary_pos_rate = canary_sub["label"].mean() if len(canary_sub) > 0 else 0
        non_canary_pos_rate = non_canary_sub["label"].mean() if len(non_canary_sub) > 0 else 0
        print(f"  {split_name}: canary pos_rate={canary_pos_rate:.3f}, "
              f"non-canary pos_rate={non_canary_pos_rate:.3f}")

    # Verify disjoint canary name pools between A and B
    a_names = {c["name"] for c in canary_metadata["canaries"] if c["split"] == "A_train"}
    b_names = {c["name"] for c in canary_metadata["canaries"] if c["split"] == "B_train"}
    overlap = a_names & b_names
    if overlap:
        print(f"  ⚠️  WARNING: {len(overlap)} canary names shared between A_train and B_train")
    else:
        print(f"  ✓ Canary name pools are disjoint between A_train and B_train")

    return df, canary_metadata


# =========================================================================== #
# Shared helpers
# =========================================================================== #

def _load_truncation_tokenizer():
    """Load a tokenizer for truncating texts to MAX_SEQ_LEN tokens.

    Tries the project's primary model tokenizer first; falls back to a
    simple character-based truncation proxy if the gated model is not
    accessible.
    """
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(
            config.MODELS["1b"], trust_remote_code=True,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok
    except Exception:
        # Gated model not accessible — fall back to GPT-2 tokenizer
        # (similar BPE vocab size; good enough for length truncation)
        print("  (Llama tokenizer unavailable — falling back to gpt2 tokenizer)")
        tok = AutoTokenizer.from_pretrained("gpt2")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok


def truncate_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Truncate the ``text`` column to ≤ MAX_SEQ_LEN tokens in-place."""
    print(f"\nTruncating texts to ≤{config.MAX_SEQ_LEN} tokens…")
    tok = _load_truncation_tokenizer()

    def _trunc(text: str) -> str:
        ids = tok.encode(
            text, add_special_tokens=False,
            truncation=True, max_length=config.MAX_SEQ_LEN,
        )
        return tok.decode(ids, skip_special_tokens=True)

    df["text"] = df["text"].apply(_trunc)
    return df


def create_splits(
    df: pd.DataFrame,
    seed: int,
    dry_run: bool,
    split_sizes_override: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Create 5 disjoint stratified splits and return the annotated DF.

    Parameters
    ----------
    split_sizes_override : dict, optional
        If provided, use these split sizes instead of ``config.SPLIT_SIZES``.
    """
    print("\nCreating disjoint splits…")
    actual_total = len(df)

    ref_split_sizes = split_sizes_override or config.SPLIT_SIZES
    ref_total = sum(ref_split_sizes.values())

    if dry_run:
        split_sizes = {k: min(20, actual_total // 5) for k in ref_split_sizes}
    else:
        scale = actual_total / ref_total
        split_sizes = {
            k: max(10, int(v * scale))
            for k, v in ref_split_sizes.items()
        }
        total_needed = sum(split_sizes.values())
        if total_needed > actual_total:
            scale = actual_total / total_needed
            split_sizes = {
                k: max(10, int(v * scale))
                for k, v in split_sizes.items()
            }

    labels = df["label"].values

    # Check if stratification is safe: every class needs at least 2 examples
    # per split.  For multiclass with rare categories, fall back to
    # unstratified splitting rather than crashing.
    n_classes = len(np.unique(labels))
    min_class_count = min(np.bincount(labels.astype(int)))
    n_splits = len(split_sizes)  # 5
    can_stratify = min_class_count >= n_splits * 2
    if not can_stratify:
        print(f"  ⚠️  Smallest class has only {min_class_count} examples "
              f"across {n_classes} classes — falling back to unstratified splits")

    def _safe_split(indices, test_size, strat_labels, rs):
        """train_test_split with stratification fallback."""
        if can_stratify:
            return train_test_split(
                indices, test_size=test_size,
                stratify=strat_labels, random_state=rs,
            )
        return train_test_split(
            indices, test_size=test_size, random_state=rs,
        )

    # 1) attack_eval
    remaining_idx, eval_idx = _safe_split(
        np.arange(len(df)),
        test_size=split_sizes["attack_eval"],
        strat_labels=labels,
        rs=seed,
    )
    # 2) attack_cal
    remaining_labels = labels[remaining_idx]
    remaining_idx, cal_idx = _safe_split(
        remaining_idx,
        test_size=split_sizes["attack_cal"],
        strat_labels=remaining_labels,
        rs=seed,
    )
    # 3) val
    remaining_labels = labels[remaining_idx]
    remaining_idx, val_idx = _safe_split(
        remaining_idx,
        test_size=split_sizes["val"],
        strat_labels=remaining_labels,
        rs=seed,
    )
    # 4) A_train / B_train
    remaining_labels = labels[remaining_idx]
    a_size = split_sizes["A_train"]
    b_size = split_sizes["B_train"]
    if a_size + b_size > len(remaining_idx):
        a_size = len(remaining_idx) // 2
        b_size = len(remaining_idx) - a_size

    a_idx, b_idx = _safe_split(
        remaining_idx,
        test_size=b_size,
        strat_labels=remaining_labels,
        rs=seed,
    )

    df["split"] = ""
    df.loc[df.index[a_idx], "split"] = "A_train"
    df.loc[df.index[b_idx], "split"] = "B_train"
    df.loc[df.index[val_idx], "split"] = "val"
    df.loc[df.index[cal_idx], "split"] = "attack_cal"
    df.loc[df.index[eval_idx], "split"] = "attack_eval"

    df = df[df["split"] != ""].reset_index(drop=True)

    # Verify disjointness
    a_orig = set(df[df["split"] == "A_train"]["original_index"])
    b_orig = set(df[df["split"] == "B_train"]["original_index"])
    overlap = a_orig & b_orig
    assert len(overlap) == 0, (
        f"A_train and B_train overlap on {len(overlap)} examples!"
    )
    return df


def print_summary(df: pd.DataFrame, output_file: Path) -> None:
    """Print a human-readable summary of the prepared dataset."""
    tok = _load_truncation_tokenizer()

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Total examples: {len(df)}")
    print(f"Output file: {output_file}")

    for split_name in ["A_train", "B_train", "val", "attack_cal", "attack_eval"]:
        sub = df[df["split"] == split_name]
        n = len(sub)
        pos = sub["label"].sum()
        neg = n - pos
        rate = pos / n if n > 0 else 0
        print(
            f"  {split_name:15s}: {n:6d} total | "
            f"{pos:5d} pos | {neg:5d} neg | pos_rate={rate:.3f}"
        )

    text_lengths = df["text"].str.len()
    print(
        f"\nText length (chars): mean={text_lengths.mean():.0f}, "
        f"median={text_lengths.median():.0f}, "
        f"min={text_lengths.min()}, max={text_lengths.max()}"
    )

    token_lengths = df["text"].apply(
        lambda t: len(tok.encode(t, add_special_tokens=False))
    )
    print(
        f"Text length (tokens): mean={token_lengths.mean():.0f}, "
        f"median={token_lengths.median():.0f}, "
        f"min={token_lengths.min()}, max={token_lengths.max()}"
    )

    label_counts = Counter(df["label"])
    print(f"\nOverall label distribution: {dict(label_counts)}")

    a_orig = set(df[df["split"] == "A_train"]["original_index"])
    b_orig = set(df[df["split"] == "B_train"]["original_index"])
    print(f"A_train ∩ B_train overlap: {len(a_orig & b_orig)} (should be 0)")
    print("=" * 70)


def save_dataset_metadata(output_dir: Path, dataset: str, **extra) -> None:
    """Write a small ``metadata.json`` alongside the parquet.

    Downstream scripts (membership_inference, analyze) read this to
    select the correct label names and include the dataset in plot
    titles.
    """
    meta = {"dataset": dataset}
    meta.update(extra)
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to {meta_path}")


# =========================================================================== #
# ToxicChat pipeline  (default)
# =========================================================================== #

def prepare_toxic_chat(args: argparse.Namespace) -> None:
    """
    Load lmsys/toxic-chat, merge toxicity + jailbreaking into a single
    positive label, balance 50/50, truncate, split, and save.
    """
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = config.DATASET_IDS["toxic-chat"]
    hf_config = config.TOXICCHAT_CONFIG

    print(f"\nDataset: {dataset_id} (config={hf_config})")
    print(f"Positive label: toxicity==1 OR jailbreaking==1")
    print(f"Target balance: {args.pos_neg_ratio:.0%} positive")

    # ----- Load both HF splits and merge into one pool -----
    print("\nDownloading ToxicChat…")
    ds_train = load_dataset(dataset_id, hf_config, split="train")
    ds_test = load_dataset(dataset_id, hf_config, split="test")

    from datasets import concatenate_datasets
    ds = concatenate_datasets([ds_train, ds_test])
    print(f"Total rows loaded: {len(ds):,}")

    if args.dry_run:
        ds = ds.select(range(min(config.DRY_RUN_SIZE, len(ds))))
        print(f"Dry-run: using first {len(ds)} rows")

    # ----- Schema exploration -----
    row = ds[0]
    print("\n--- ToxicChat schema ---")
    for k in sorted(row.keys()):
        v = row[k]
        print(f"  {k}: {type(v).__name__}  →  {repr(v)[:100]}")
    print()

    # ----- Build records -----
    positives: list[dict] = []
    negatives: list[dict] = []

    for i, row in enumerate(tqdm(ds, desc="Processing ToxicChat")):
        text = (row.get("user_input") or "").strip()
        if not text:
            continue

        toxicity = int(row.get("toxicity", 0))
        jailbreaking = int(row.get("jailbreaking", 0))
        label = 1 if (toxicity == 1 or jailbreaking == 1) else 0

        meta = {"toxicity": toxicity, "jailbreaking": jailbreaking}
        oai_mod = row.get("openai_moderation")
        if oai_mod and isinstance(oai_mod, str):
            meta["openai_moderation_raw"] = oai_mod

        entry = {
            "text": text,
            "label": label,
            "original_index": i,
            "moderation_scores": json.dumps(meta),
        }

        if label == 1:
            positives.append(entry)
        else:
            negatives.append(entry)

    print(f"\n--- Collection summary ---")
    print(f"Positives (toxic OR jailbreak): {len(positives):,}")
    print(f"Negatives (safe):               {len(negatives):,}")

    if len(positives) < config.MIN_POSITIVES_WARN:
        print(
            f"\n⚠️  WARNING: Only {len(positives)} positives "
            f"(< {config.MIN_POSITIVES_WARN})."
        )
        if len(positives) == 0:
            print("   Cannot proceed with 0 positives. Exiting.")
            sys.exit(1)

    # ----- Balance 50 / 50 -----
    rng = np.random.RandomState(args.seed)
    n_pos = len(positives)
    target_neg = int(n_pos * (1 - args.pos_neg_ratio) / args.pos_neg_ratio)
    n_neg = min(len(negatives), target_neg)

    target_total = args.total_size
    if target_total and (n_pos + n_neg) > target_total:
        n_pos = int(target_total * args.pos_neg_ratio)
        n_neg = target_total - n_pos

    n_pos = min(n_pos, len(positives))
    n_neg = min(n_neg, len(negatives))

    pos_idx = rng.choice(len(positives), size=n_pos, replace=False)
    neg_idx = rng.choice(len(negatives), size=n_neg, replace=False)

    sampled = [positives[i] for i in pos_idx] + [negatives[i] for i in neg_idx]
    df = pd.DataFrame(sampled)

    actual_total = len(df)
    print(
        f"\nBalanced dataset: {actual_total} total "
        f"({n_pos} positive, {n_neg} negative)"
    )
    print(f"Positive rate: {n_pos / actual_total:.3f}")

    # ----- Truncate / split / save -----
    df = truncate_texts(df)
    df = create_splits(df, args.seed, args.dry_run)
    df = _maybe_insert_canaries(df, args, output_dir)

    output_file = output_dir / config.DATA_FILE
    df.to_parquet(output_file, index=False)
    print(f"\nSaved to {output_file}")
    save_dataset_metadata(output_dir, "toxic-chat")
    print_summary(df, output_file)


# =========================================================================== #
# WildChat pipeline  (fallback)
# =========================================================================== #

def explore_schema(dataset_id: str) -> None:
    """Load 10 rows and print the full schema of openai_moderation."""
    print("=" * 70)
    print(f"SCHEMA EXPLORATION — loading 10 rows from {dataset_id}")
    print("=" * 70)

    ds = load_dataset(dataset_id, split="train", streaming=True)
    samples = list(ds.take(10))

    if not samples:
        print("ERROR: could not load any samples.")
        return

    row = samples[0]
    print("\n--- Top-level columns ---")
    for k in sorted(row.keys()):
        v = row[k]
        tp = type(v).__name__
        if isinstance(v, list):
            tp += f"[len={len(v)}]"
        print(f"  {k}: {tp}")

    # conversation structure
    print("\n--- conversation[0] keys ---")
    if row.get("conversation") and len(row["conversation"]) > 0:
        for k, v in row["conversation"][0].items():
            print(f"  {k}: {type(v).__name__}  →  {repr(v)[:120]}")

    # openai_moderation structure
    print("\n--- openai_moderation field ---")
    mod = row.get("openai_moderation")
    if mod is None:
        print("  Field is None")
    elif isinstance(mod, list):
        print(f"  Type: list, length={len(mod)}")
        if len(mod) > 0:
            first = mod[0]
            if isinstance(first, dict):
                print(f"  openai_moderation[0] keys: {sorted(first.keys())}")
                if "categories" in first:
                    print(f"  categories keys: {sorted(first['categories'].keys())}")
                    print(f"  categories sample: {first['categories']}")
                if "category_scores" in first:
                    print(f"  category_scores keys: {sorted(first['category_scores'].keys())}")
                    print(f"  category_scores sample:")
                    for cat, score in sorted(first["category_scores"].items()):
                        print(f"    {cat}: {score}")
            else:
                print(f"  openai_moderation[0] type: {type(first).__name__}")
                print(f"  openai_moderation[0] value: {repr(first)[:200]}")
    else:
        print(f"  Type: {type(mod).__name__}")
        print(f"  Value: {repr(mod)[:500]}")

    print("\n--- Full first example for reference ---")
    print(f"  language: {row.get('language')}")
    print(f"  toxic: {row.get('toxic')}")
    print(f"  conversation turns: {len(row.get('conversation', []))}")
    if row.get("conversation"):
        print(f"  first turn role: {row['conversation'][0].get('role')}")
        print(f"  first turn content[:200]: {row['conversation'][0].get('content', '')[:200]}")
    print("=" * 70)


# =========================================================================== #
# Parsing helpers
# =========================================================================== #

def extract_first_user_turn(conversation: list[dict]) -> str | None:
    """Return the content of the first user turn, or None."""
    if not conversation:
        return None
    for turn in conversation:
        if turn.get("role") == "user":
            content = turn.get("content", "").strip()
            return content if content else None
    return None


def extract_moderation_scores(
    openai_moderation: list | None,
    conversation: list[dict] | None,
    categories: list[str],
) -> dict[str, float] | None:
    """
    Extract moderation scores for the first user turn.

    openai_moderation is a list with one entry per utterance (user and
    assistant interleaved).  We need the entry corresponding to the
    first user turn.
    """
    if not openai_moderation or not conversation:
        return None

    # Find the index of the first user turn in the conversation list
    first_user_idx = None
    for i, turn in enumerate(conversation):
        if turn.get("role") == "user":
            first_user_idx = i
            break

    if first_user_idx is None:
        return None

    # The moderation list should be aligned with the conversation list
    if first_user_idx >= len(openai_moderation):
        return None

    mod_entry = openai_moderation[first_user_idx]
    if not isinstance(mod_entry, dict) or not mod_entry:
        return None

    scores_dict = mod_entry.get("category_scores")
    if not scores_dict or not isinstance(scores_dict, dict):
        return None

    result = {}
    for cat in categories:
        score = scores_dict.get(cat)
        if score is not None:
            result[cat] = float(score)
        else:
            result[cat] = 0.0
    return result


def apply_label(
    scores: dict[str, float],
    thresholds: dict[str, float],
) -> int:
    """Return 1 (harmful) if any category score exceeds its threshold."""
    for cat, thresh in thresholds.items():
        if scores.get(cat, 0.0) > thresh:
            return 1
    return 0


# =========================================================================== #
# Main data-preparation pipeline
# =========================================================================== #

def prepare_wildchat(args: argparse.Namespace) -> None:
    """WildChat pipeline — stream, threshold moderation scores, split."""
    set_seed(args.seed)

    dataset_id = config.DATASET_IDS[args.dataset]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse category/threshold args
    categories = [c.strip() for c in args.categories.split(",")]
    thresholds_list = [float(t) for t in args.thresholds.split(",")]
    if len(categories) != len(thresholds_list):
        print("ERROR: --categories and --thresholds must have the same number of entries.")
        sys.exit(1)
    thresholds = dict(zip(categories, thresholds_list))

    print(f"\nDataset: {dataset_id}")
    print(f"Categories & thresholds: {thresholds}")
    print(f"Target total size: {args.total_size}")
    print(f"Positive fraction: {args.pos_neg_ratio}")

    # ----- Schema exploration -----
    explore_schema(dataset_id)

    # ----- Stream through dataset -----
    print("\nLoading and processing dataset (streaming)…")
    ds = load_dataset(dataset_id, split="train", streaming=True)

    positives: list[dict] = []
    negatives: list[dict] = []
    skipped_lang = 0
    skipped_no_turn = 0
    skipped_no_mod = 0
    total_seen = 0
    max_rows = config.DRY_RUN_SIZE if args.dry_run else None  # None = scan everything

    # How many negatives we might need (upper bound)
    target_pos = int(args.total_size * args.pos_neg_ratio)
    target_neg = args.total_size - target_pos
    # Collect more negatives than needed so we can sample later
    neg_collect_limit = target_neg * 3  # over-collect, sample later

    for row in tqdm(ds, desc="Scanning WildChat", total=max_rows):
        total_seen += 1
        if max_rows and total_seen > max_rows:
            break

        # English only
        if row.get("language", "").lower() not in ("english", "en"):
            skipped_lang += 1
            continue

        # Extract first user turn
        conversation = row.get("conversation", [])
        text = extract_first_user_turn(conversation)
        if not text:
            skipped_no_turn += 1
            continue

        # Extract moderation scores
        scores = extract_moderation_scores(
            row.get("openai_moderation"), conversation, categories
        )
        if scores is None:
            skipped_no_mod += 1
            continue

        label = apply_label(scores, thresholds)
        entry = {
            "text": text,
            "label": label,
            "original_index": total_seen - 1,
            "moderation_scores": json.dumps(scores),
        }

        if label == 1:
            positives.append(entry)
        else:
            if len(negatives) < neg_collect_limit:
                negatives.append(entry)

        # Early stop: we have enough of both classes
        if len(positives) >= target_pos * 2 and len(negatives) >= neg_collect_limit:
            print(f"\nCollected enough examples after scanning {total_seen} rows. Stopping early.")
            break

    print(f"\n--- Collection summary ---")
    print(f"Total rows scanned: {total_seen:,}")
    print(f"Skipped (non-English): {skipped_lang:,}")
    print(f"Skipped (no user turn): {skipped_no_turn:,}")
    print(f"Skipped (no moderation): {skipped_no_mod:,}")
    print(f"Positives collected: {len(positives):,}")
    print(f"Negatives collected: {len(negatives):,}")

    # ----- Check positive count -----
    if len(positives) < config.MIN_POSITIVES_WARN:
        print(f"\n⚠️  WARNING: Only {len(positives)} positive examples found (< {config.MIN_POSITIVES_WARN}).")
        print("   Consider using --dataset full (the gated version with toxic conversations),")
        print("   lowering thresholds, or adding more categories.")
        if len(positives) == 0:
            print("   Cannot proceed with 0 positives. Exiting.")
            sys.exit(1)

    # ----- Sample to target sizes -----
    n_pos = min(len(positives), target_pos)
    n_neg = min(len(negatives), target_neg)

    # Adjust if not enough positives
    if n_pos < target_pos:
        print(f"\nAdjusting: only {n_pos} positives available (target was {target_pos}).")
        n_neg = min(len(negatives), int(n_pos / args.pos_neg_ratio * (1 - args.pos_neg_ratio)))

    rng = np.random.RandomState(args.seed)
    pos_indices = rng.choice(len(positives), size=n_pos, replace=False)
    neg_indices = rng.choice(len(negatives), size=n_neg, replace=False)

    sampled = [positives[i] for i in pos_indices] + [negatives[i] for i in neg_indices]
    df = pd.DataFrame(sampled)

    actual_total = len(df)
    print(f"\nSampled dataset: {actual_total} total ({n_pos} positive, {n_neg} negative)")
    print(f"Positive rate: {n_pos / actual_total:.3f}")

    # ----- Truncate / split / save -----
    df = truncate_texts(df)
    df = create_splits(df, args.seed, args.dry_run)
    df = _maybe_insert_canaries(df, args, output_dir)

    output_file = output_dir / config.DATA_FILE
    df.to_parquet(output_file, index=False)
    print(f"\nSaved to {output_file}")
    save_dataset_metadata(output_dir, args.dataset)
    print_summary(df, output_file)


# =========================================================================== #
# BeaverTails pipeline
# =========================================================================== #

def prepare_beavertails(args: argparse.Namespace) -> None:
    """
    Load PKU-Alignment/BeaverTails, label unsafe/safe using ``is_safe``,
    optionally filter to specific harm categories, balance 50/50,
    truncate, split, and save.
    """
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = config.DATASET_IDS["beavertails"]

    # Parse optional category filter
    selected_categories: list[str] | None = None
    if args.bt_categories:
        selected_categories = [c.strip() for c in args.bt_categories.split(",")]
        unknown = set(selected_categories) - set(config.BEAVERTAILS_CATEGORIES)
        if unknown:
            print(f"ERROR: Unknown BeaverTails categories: {unknown}")
            print(f"Valid categories: {config.BEAVERTAILS_CATEGORIES}")
            sys.exit(1)

    label_mode = getattr(args, "label_mode", "binary")

    print(f"\nDataset: {dataset_id}")
    print(f"Label mode: {label_mode}")
    if label_mode == "multiclass":
        n_classes = config.get_num_labels("beavertails", "multiclass")
        print(f"Multi-class: {n_classes} classes (14 harm categories + safe)")
    elif selected_categories:
        print(f"Category filter: {selected_categories}")
        print(f"Positive label: any selected category is True")
    else:
        print(f"Positive label: is_safe == False (any of 14 categories)")
    print(f"Target balance: {args.pos_neg_ratio:.0%} positive")

    # Decide target total and split sizes
    total_size = args.total_size
    if total_size == config.TOTAL_SIZE:
        # User didn't override → use BeaverTails default
        total_size = config.BEAVERTAILS_TOTAL_SIZE

    # ----- Load both HF splits and merge into one pool -----
    print("\nDownloading BeaverTails…")
    from datasets import concatenate_datasets

    ds_train = load_dataset(dataset_id, split="330k_train")
    ds_test = load_dataset(dataset_id, split="330k_test")
    ds = concatenate_datasets([ds_train, ds_test])
    print(f"Total rows loaded: {len(ds):,}")

    if args.dry_run:
        ds = ds.select(range(min(config.DRY_RUN_SIZE, len(ds))))
        print(f"Dry-run: using first {len(ds)} rows")

    # ----- Schema exploration -----
    row0 = ds[0]
    print("\n--- BeaverTails schema ---")
    for k in sorted(row0.keys()):
        v = row0[k]
        print(f"  {k}: {type(v).__name__}  →  {repr(v)[:120]}")
    print()

    # ----- Build records -----
    positives: list[dict] = []
    negatives: list[dict] = []

    for i, row in enumerate(tqdm(ds, desc="Processing BeaverTails")):
        text = (row.get("prompt") or "").strip()
        if not text:
            continue

        # Category annotations (dict of 14 bools)
        category = row.get("category", {})
        if not isinstance(category, dict):
            category = {}

        # Determine label
        label_mode = getattr(args, "label_mode", "binary")
        if label_mode == "multiclass":
            # Multi-class: label = index of primary category (1-14) or 0 (safe)
            if row.get("is_safe", True):
                label = 0  # safe
            else:
                label = 0  # default safe; overridden below if category found
                for cat_idx, cat in enumerate(config.BEAVERTAILS_CATEGORIES):
                    if category.get(cat, False):
                        label = cat_idx + 1  # categories are 1-indexed
                        break
        elif selected_categories:
            # Custom filter: positive if *any* selected category is True
            label = 1 if any(category.get(c, False) for c in selected_categories) else 0
        else:
            # Default: use the overall is_safe flag
            is_safe = row.get("is_safe", True)
            label = 0 if is_safe else 1

        meta = {
            "is_safe": bool(row.get("is_safe", True)),
            "categories": {k: bool(v) for k, v in category.items()},
        }
        if selected_categories:
            meta["category_filter"] = selected_categories

        # Determine primary category (first True in config order)
        if label == 1:
            primary_cat = "unknown"
            for cat in config.BEAVERTAILS_CATEGORIES:
                if category.get(cat, False):
                    primary_cat = cat
                    break
        else:
            primary_cat = "safe"

        entry = {
            "text": text,
            "label": label,
            "original_index": i,
            "moderation_scores": json.dumps(meta),
            "bt_category": primary_cat,
        }

        if label_mode == "multiclass":
            # In multiclass mode, positives are labels 1-14, negatives are label 0
            if label > 0:
                positives.append(entry)
            else:
                negatives.append(entry)
        else:
            if label == 1:
                positives.append(entry)
            else:
                negatives.append(entry)

    print(f"\n--- Collection summary ---")
    print(f"Positives (unsafe): {len(positives):,}")
    print(f"Negatives (safe):   {len(negatives):,}")

    if len(positives) < config.MIN_POSITIVES_WARN:
        print(
            f"\n⚠️  WARNING: Only {len(positives)} positives "
            f"(< {config.MIN_POSITIVES_WARN})."
        )
        if len(positives) == 0:
            print("   Cannot proceed with 0 positives. Exiting.")
            sys.exit(1)

    # ----- Balance 50 / 50 -----
    rng = np.random.RandomState(args.seed)
    n_pos = len(positives)
    target_neg = int(n_pos * (1 - args.pos_neg_ratio) / args.pos_neg_ratio)
    n_neg = min(len(negatives), target_neg)

    if total_size and (n_pos + n_neg) > total_size:
        n_pos = int(total_size * args.pos_neg_ratio)
        n_neg = total_size - n_pos

    n_pos = min(n_pos, len(positives))
    n_neg = min(n_neg, len(negatives))

    pos_idx = rng.choice(len(positives), size=n_pos, replace=False)
    neg_idx = rng.choice(len(negatives), size=n_neg, replace=False)

    sampled = [positives[i] for i in pos_idx] + [negatives[i] for i in neg_idx]
    df = pd.DataFrame(sampled)

    actual_total = len(df)
    print(
        f"\nBalanced dataset: {actual_total} total "
        f"({n_pos} positive, {n_neg} negative)"
    )
    print(f"Positive rate: {n_pos / actual_total:.3f}")

    # ----- Truncate / split / save -----
    df = truncate_texts(df)

    # Add token counts (post-truncation) for breakdown_by_token_length
    print("Counting tokens…")
    tok = _load_truncation_tokenizer()
    df["token_count"] = df["text"].apply(lambda t: _count_tokens_exact(t, tok))
    print(f"  Token count range: {df['token_count'].min()} – {df['token_count'].max()} "
          f"(mean: {df['token_count'].mean():.0f})")

    # Use BeaverTails split sizes unless user overrode --total_size
    split_sizes = None
    if args.total_size == config.TOTAL_SIZE:
        split_sizes = config.BEAVERTAILS_SPLIT_SIZES
    df = create_splits(df, args.seed, args.dry_run, split_sizes_override=split_sizes)
    df = _maybe_insert_canaries(df, args, output_dir)

    output_file = output_dir / config.DATA_FILE
    df.to_parquet(output_file, index=False)
    print(f"\nSaved to {output_file}")
    label_mode = getattr(args, "label_mode", "binary")
    save_dataset_metadata(
        output_dir, "beavertails",
        category_filter=selected_categories or "all",
        label_mode=label_mode,
    )
    print_summary(df, output_file)


# =========================================================================== #
# WildChat language pipeline  (proxy task — English vs non-English)
# =========================================================================== #

def prepare_wildchat_language(args: argparse.Namespace) -> None:
    """WildChat language pipeline — binary English vs non-English label.

    Instead of toxicity, we use the dataset's ``language`` column as the
    classification target.  This gives us a large, balanced dataset for
    membership inference experiments while sidestepping the small sample
    size of rare toxicity labels.

    The raw ``language`` string is preserved as an extra column in the
    output parquet so downstream analyses can break down MIA results by
    individual language.
    """
    set_seed(args.seed)

    dataset_id = config.DATASET_IDS[args.dataset]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the larger language-mode split sizes
    total_size = args.total_size
    if total_size == config.TOTAL_SIZE:
        # User didn't override → use the language-mode default
        total_size = config.WILDCHAT_LANGUAGE_TOTAL_SIZE

    print(f"\nDataset: {dataset_id}")
    print(f"Mode: LANGUAGE (English=1 vs non-English=0)")
    print(f"Target total size: {total_size}")
    print(f"Positive fraction: {args.pos_neg_ratio}")

    # ----- Schema exploration -----
    explore_schema(dataset_id)

    # ----- Stream through dataset -----
    print("\nLoading and processing dataset (streaming)…")
    ds = load_dataset(dataset_id, split="train", streaming=True)

    english_samples: list[dict] = []
    non_english_samples: list[dict] = []
    skipped_no_turn = 0
    total_seen = 0
    max_rows = config.DRY_RUN_SIZE if args.dry_run else None

    target_english = int(total_size * args.pos_neg_ratio)
    target_non_english = total_size - target_english
    # Over-collect so we can sample later
    collect_limit = max(target_english, target_non_english) * 3

    for row in tqdm(ds, desc="Scanning WildChat (language mode)", total=max_rows):
        total_seen += 1
        if max_rows and total_seen > max_rows:
            break

        # Extract first user turn
        conversation = row.get("conversation", [])
        text = extract_first_user_turn(conversation)
        if not text:
            skipped_no_turn += 1
            continue

        raw_language = (row.get("language") or "").strip()
        if not raw_language:
            continue

        is_english = raw_language.lower() in ("english", "en")
        label = 1 if is_english else 0

        entry = {
            "text": text,
            "label": label,
            "original_index": total_seen - 1,
            "moderation_scores": json.dumps({"language": raw_language}),
            "language": raw_language,
        }

        if is_english:
            if len(english_samples) < collect_limit:
                english_samples.append(entry)
        else:
            if len(non_english_samples) < collect_limit:
                non_english_samples.append(entry)

        # Early stop when we have enough of both classes
        if (len(english_samples) >= collect_limit
                and len(non_english_samples) >= collect_limit):
            print(f"\nCollected enough examples after scanning {total_seen} rows. Stopping early.")
            break

    print(f"\n--- Collection summary ---")
    print(f"Total rows scanned: {total_seen:,}")
    print(f"Skipped (no user turn): {skipped_no_turn:,}")
    print(f"English collected: {len(english_samples):,}")
    print(f"Non-English collected: {len(non_english_samples):,}")

    if len(english_samples) < config.MIN_POSITIVES_WARN:
        print(f"\n⚠️  WARNING: Only {len(english_samples)} English examples found.")
    if len(non_english_samples) < config.MIN_POSITIVES_WARN:
        print(f"\n⚠️  WARNING: Only {len(non_english_samples)} non-English examples found.")
    if len(english_samples) == 0 or len(non_english_samples) == 0:
        print("   Cannot proceed with 0 examples in either class. Exiting.")
        sys.exit(1)

    # ----- Sample to target sizes -----
    n_eng = min(len(english_samples), target_english)
    n_non = min(len(non_english_samples), target_non_english)

    # Adjust if one side is short
    if n_eng < target_english:
        print(f"\nAdjusting: only {n_eng} English available (target was {target_english}).")
        n_non = min(len(non_english_samples), n_eng)  # keep balanced
    if n_non < target_non_english:
        print(f"\nAdjusting: only {n_non} non-English available (target was {target_non_english}).")
        n_eng = min(len(english_samples), n_non)

    rng = np.random.RandomState(args.seed)
    eng_idx = rng.choice(len(english_samples), size=n_eng, replace=False)
    non_idx = rng.choice(len(non_english_samples), size=n_non, replace=False)

    sampled = ([english_samples[i] for i in eng_idx]
               + [non_english_samples[i] for i in non_idx])
    df = pd.DataFrame(sampled)

    actual_total = len(df)
    print(f"\nSampled dataset: {actual_total} total ({n_eng} English, {n_non} non-English)")
    print(f"English rate: {n_eng / actual_total:.3f}")

    # Show top languages in non-English subset
    lang_counts = df[df["label"] == 0]["language"].value_counts().head(15)
    print(f"\nTop non-English languages:\n{lang_counts.to_string()}")

    # ----- Truncate / split / save -----
    df = truncate_texts(df)

    # Use language-mode split sizes unless user overrode --total_size
    split_sizes = None
    if args.total_size == config.TOTAL_SIZE:
        split_sizes = config.WILDCHAT_LANGUAGE_SPLIT_SIZES
    df = create_splits(df, args.seed, args.dry_run, split_sizes_override=split_sizes)

    # ----- Per-split language breakdown -----
    print("\n" + "=" * 70)
    print("LANGUAGE BREAKDOWN BY SPLIT")
    print("=" * 70)
    for split_name in ["A_train", "B_train", "val", "attack_cal", "attack_eval"]:
        sub = df[df["split"] == split_name]
        n = len(sub)
        if n == 0:
            continue
        n_eng = (sub["label"] == 1).sum()
        n_non = (sub["label"] == 0).sum()
        print(f"\n  {split_name} ({n} total — {n_eng} English, {n_non} non-English)")
        print(f"  {'─' * 50}")

        # Language distribution for non-English examples in this split
        non_eng = sub[sub["label"] == 0]
        if len(non_eng) > 0:
            lang_counts = non_eng["language"].value_counts()
            top_n = min(15, len(lang_counts))
            top_langs = lang_counts.head(top_n)
            other_count = lang_counts.iloc[top_n:].sum() if len(lang_counts) > top_n else 0
            n_unique = lang_counts.nunique()

            print(f"  Non-English languages ({n_unique} unique):")
            for lang, count in top_langs.items():
                pct = count / len(non_eng) * 100
                print(f"    {lang:25s}: {count:5d}  ({pct:5.1f}%)")
            if other_count > 0:
                pct = other_count / len(non_eng) * 100
                print(f"    {'(other)':25s}: {other_count:5d}  ({pct:5.1f}%)")

    # Cross-split language consistency check
    print(f"\n  {'─' * 50}")
    print("  Cross-split language overlap (non-English):")
    split_langs = {}
    for split_name in ["A_train", "B_train", "val", "attack_cal", "attack_eval"]:
        sub = df[(df["split"] == split_name) & (df["label"] == 0)]
        split_langs[split_name] = set(sub["language"].unique()) if len(sub) > 0 else set()

    # Report languages unique to training splits vs attack splits
    train_langs = split_langs.get("A_train", set()) | split_langs.get("B_train", set())
    attack_langs = split_langs.get("attack_cal", set()) | split_langs.get("attack_eval", set())
    train_only = train_langs - attack_langs
    attack_only = attack_langs - train_langs
    shared = train_langs & attack_langs
    print(f"    Languages in both train & attack splits: {len(shared)}")
    if train_only:
        print(f"    Languages ONLY in train splits:          {len(train_only)}  {sorted(train_only)}")
    if attack_only:
        print(f"    Languages ONLY in attack splits:         {len(attack_only)}  {sorted(attack_only)}")
    print("=" * 70)

    df = _maybe_insert_canaries(df, args, output_dir)

    output_file = output_dir / config.DATA_FILE
    df.to_parquet(output_file, index=False)
    print(f"\nSaved to {output_file}")
    save_dataset_metadata(output_dir, args.dataset)
    print_summary(df, output_file)


# =========================================================================== #
# XGuard multi-turn pipeline  (long-context MIA experiment)
# =========================================================================== #

def _count_tokens_exact(text: str, tokenizer) -> int:
    """Return the exact token count for *text* using the given tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def _xguard_role(raw_role: str) -> str:
    """Normalise XGuard role names (``human``/``gpt``) to standard roles."""
    return "user" if raw_role == "human" else "assistant"


def _format_turns_as_text(turns: list[dict]) -> str:
    """Concatenate turns into a plain-text representation for display/analysis."""
    parts = []
    for t in turns:
        role = t.get("role", "unknown")
        content = t.get("content", "")
        parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts)


def prepare_xguard_multiturn(args: argparse.Namespace) -> None:
    """
    XGuard + WildChat multi-turn pipeline — long-context MIA experiment.

    Harmful source : marslabucla/XGuard-Train  (multi-turn jailbreak trajectories)
    Benign  source : allenai/WildChat-1M        (multi-turn benign conversations)

    Only conversations whose token count ≥ ``XGUARD_MIN_TOKEN_RATIO * XGUARD_MAX_SEQ_LEN``
    are kept, so every sample fills (or nearly fills) the context window.

    Label:  1 = harmful trajectory (XGuard),  0 = benign conversation (WildChat)
    """
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    min_tokens = int(config.XGUARD_MAX_SEQ_LEN * config.XGUARD_MIN_TOKEN_RATIO)
    max_seq_len = config.XGUARD_MAX_SEQ_LEN

    print(f"\n{'='*70}")
    print("  XGUARD MULTI-TURN — Long-context MIA experiment")
    print(f"{'='*70}")
    print(f"  Harmful source : {config.DATASET_IDS['xguard-multiturn']}")
    print(f"  Benign  source : {config.XGUARD_BENIGN_DATASET_ID}")
    print(f"  Context window : {max_seq_len} tokens")
    print(f"  Min tokens     : {min_tokens} ({config.XGUARD_MIN_TOKEN_RATIO:.0%} of context)")
    print(f"  Target balance : 50/50")

    # Load tokenizer for accurate token counting
    tok = _load_truncation_tokenizer()

    # --------------------------------------------------------------------- #
    #  Harmful source: XGuard-Train
    # --------------------------------------------------------------------- #
    print(f"\n--- Loading XGuard-Train ---")
    xguard_ds = load_dataset(
        config.DATASET_IDS["xguard-multiturn"], split="train",
    )
    print(f"  Total XGuard conversations: {len(xguard_ds):,}")

    if args.dry_run:
        xguard_ds = xguard_ds.select(range(min(500, len(xguard_ds))))
        print(f"  Dry-run: using first {len(xguard_ds)} rows")

    harmful_records: list[dict] = []
    xguard_short = 0

    for i, row in enumerate(tqdm(xguard_ds, desc="Processing XGuard")):
        conversations_raw = row.get("conversations", [])
        if not conversations_raw:
            continue

        # Normalise roles: {"from": "human"/"gpt", "value": ...}
        turns = []
        for turn in conversations_raw:
            role = _xguard_role(turn.get("from", ""))
            content = (turn.get("value") or "").strip()
            if content:
                turns.append({"role": role, "content": content})

        if len(turns) < 2:
            continue

        # Compute token count on the plain-text representation
        text = _format_turns_as_text(turns)
        token_count = _count_tokens_exact(text, tok)

        if token_count < min_tokens:
            xguard_short += 1
            continue

        harmful_records.append({
            "text": text,
            "conversations": json.dumps(turns),
            "label": 1,
            "original_index": i,
            "source": "xguard",
            "num_turns": len(turns),
            "token_count": token_count,
            "moderation_scores": json.dumps({"source": "xguard"}),
        })

    print(f"  Kept {len(harmful_records):,} harmful (≥{min_tokens} tokens)")
    print(f"  Skipped {xguard_short:,} (too short)")

    # --------------------------------------------------------------------- #
    #  Benign source: WildChat multi-turn
    # --------------------------------------------------------------------- #
    print(f"\n--- Loading WildChat benign multi-turn ---")
    wildchat_ds = load_dataset(
        config.XGUARD_BENIGN_DATASET_ID, split="train", streaming=True,
    )

    benign_records: list[dict] = []
    wc_skipped_lang = 0
    wc_skipped_turns = 0
    wc_skipped_short = 0
    wc_total_seen = 0
    max_benign = len(harmful_records) * 3  # over-collect, then sample
    max_scan = 2_000 if args.dry_run else None  # None = no limit

    for row in tqdm(wildchat_ds, desc="Scanning WildChat", total=max_scan):
        wc_total_seen += 1
        if max_scan and wc_total_seen > max_scan:
            break

        # English only
        if (row.get("language") or "").lower() not in ("english", "en"):
            wc_skipped_lang += 1
            continue

        conversation = row.get("conversation", [])
        # Multi-turn: require ≥ 3 turns (at least 2 user + 1 assistant)
        if len(conversation) < 3:
            wc_skipped_turns += 1
            continue

        turns = []
        for turn in conversation:
            role = turn.get("role", "unknown")
            content = (turn.get("content") or "").strip()
            if content and role in ("user", "assistant"):
                turns.append({"role": role, "content": content})

        if len(turns) < 3:
            wc_skipped_turns += 1
            continue

        text = _format_turns_as_text(turns)
        token_count = _count_tokens_exact(text, tok)

        if token_count < min_tokens:
            wc_skipped_short += 1
            continue

        benign_records.append({
            "text": text,
            "conversations": json.dumps(turns),
            "label": 0,
            "original_index": 1_000_000 + wc_total_seen - 1,  # offset to avoid collision with XGuard indices
            "source": "wildchat",
            "num_turns": len(turns),
            "token_count": token_count,
            "moderation_scores": json.dumps({"source": "wildchat"}),
        })

        if len(benign_records) >= max_benign:
            print(f"\n  Collected enough benign ({len(benign_records)}) after scanning {wc_total_seen:,} rows")
            break

    print(f"\n  WildChat scan summary:")
    print(f"    Total scanned : {wc_total_seen:,}")
    print(f"    Skipped (lang) : {wc_skipped_lang:,}")
    print(f"    Skipped (turns): {wc_skipped_turns:,}")
    print(f"    Skipped (short): {wc_skipped_short:,}")
    print(f"    Kept benign    : {len(benign_records):,}")

    # --------------------------------------------------------------------- #
    #  Balance & merge
    # --------------------------------------------------------------------- #
    n_harmful = len(harmful_records)
    n_benign = len(benign_records)
    if n_harmful == 0 or n_benign == 0:
        print("\n  ✗ Cannot proceed — one class has zero samples. Exiting.")
        sys.exit(1)

    # Determine target total
    total_size = args.total_size
    if total_size == config.TOTAL_SIZE:
        total_size = config.XGUARD_TOTAL_SIZE

    target_per_class = total_size // 2

    rng = np.random.RandomState(args.seed)
    n_pos = min(n_harmful, target_per_class)
    n_neg = min(n_benign, n_pos)  # match to keep 50/50
    n_pos = min(n_pos, n_neg)     # ensure exact balance

    pos_idx = rng.choice(n_harmful, size=n_pos, replace=False)
    neg_idx = rng.choice(n_benign, size=n_neg, replace=False)
    sampled = ([harmful_records[i] for i in pos_idx]
               + [benign_records[i] for i in neg_idx])

    df = pd.DataFrame(sampled)
    print(f"\n  Balanced dataset: {len(df)} total "
          f"({n_pos} harmful, {n_neg} benign)")

    # Token count stats
    print(f"  Token counts — mean: {df['token_count'].mean():.0f}, "
          f"median: {df['token_count'].median():.0f}, "
          f"min: {df['token_count'].min()}, max: {df['token_count'].max()}")
    print(f"  Turn counts  — mean: {df['num_turns'].mean():.1f}, "
          f"median: {df['num_turns'].median():.0f}")

    # NOTE: We do NOT truncate here — the raw texts already meet the
    # min_tokens threshold and will be truncated to XGUARD_MAX_SEQ_LEN
    # at tokenisation time (finetune.py / membership_inference.py).

    # ----- Split -----
    split_sizes = None
    if args.total_size == config.TOTAL_SIZE:
        split_sizes = config.XGUARD_SPLIT_SIZES
    df = create_splits(df, args.seed, args.dry_run, split_sizes_override=split_sizes)
    df = _maybe_insert_canaries(df, args, output_dir)

    # ----- Save -----
    output_file = output_dir / config.DATA_FILE
    df.to_parquet(output_file, index=False)
    print(f"\n  Saved to {output_file}")
    save_dataset_metadata(
        output_dir, "xguard-multiturn",
        max_seq_len=max_seq_len,
        min_tokens=min_tokens,
    )
    print_summary(df, output_file)


# =========================================================================== #
# Psychotherapy adverse-event pipeline
# =========================================================================== #

def _download_csv(url: str, cache_path: Path) -> pd.DataFrame:
    """Download a CSV from *url* (caching to *cache_path*) and return a DataFrame."""
    import requests

    if cache_path.exists():
        print(f"  Using cached {cache_path.name}")
        return pd.read_csv(cache_path)

    print(f"  Downloading {url.split('/')[-1]} …")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(resp.content)
    return pd.read_csv(cache_path)


def _build_session_labels(adverse_df: pd.DataFrame) -> pd.DataFrame:
    """Build a (pairing_id, session_id) → binary label mapping.

    Label = 1 if **any** of the 10 adverse event types ``occurred == True``
    for that session.
    """
    # Normalise the 'occurred' column (handles string "True"/"False" or bool)
    adverse_df["occurred"] = adverse_df["occurred"].apply(
        lambda v: str(v).strip().lower() == "true"
    )

    labels = (
        adverse_df
        .groupby(["pairing_id", "session_id"])["occurred"]
        .any()
        .astype(int)
        .reset_index()
        .rename(columns={"occurred": "label"})
    )
    return labels


def _build_conversations(
    conv_df: pd.DataFrame,
) -> dict[tuple[int, int], list[dict]]:
    """Group conversation_log into per-(pairing, session) turn lists.

    Returns a dict  ``(pairing_id, session_id) -> [{"role": ..., "content": ...}, ...]``
    mapping ``"Patient"`` → ``"user"`` and ``"Therapist"`` → ``"assistant"``.
    """
    ROLE_MAP = {"Patient": "user", "Therapist": "assistant"}
    conversations: dict[tuple[int, int], list[dict]] = {}

    for (pid, sid), group in conv_df.groupby(["pairing_id", "session_id"]):
        turns = []
        for _, row in group.sort_values("turn").iterrows():
            speaker = str(row.get("speaker", "")).strip()
            role = ROLE_MAP.get(speaker)
            if role is None:
                continue  # skip unknown speakers
            content = str(row.get("message", "")).strip()
            if content:
                turns.append({"role": role, "content": content})
        if turns:
            conversations[(int(pid), int(sid))] = turns
    return conversations


# --------------------------------------------------------------------------- #
# WildChat negative augmentation for psychotherapy experiments
# --------------------------------------------------------------------------- #

def _wildchat_max_category_score(
    openai_moderation: list | None,
    category: str,
) -> float:
    """Return the max moderation score for *category* across all turns."""
    if not openai_moderation:
        return 0.0
    best = 0.0
    for entry in openai_moderation:
        if entry is None or not isinstance(entry, dict):
            continue
        cs = entry.get("category_scores")
        if cs and isinstance(cs, dict):
            best = max(best, float(cs.get(category, 0.0)))
    return best


def _sample_wildchat_negatives(
    n_needed: int,
    seed: int,
    dry_run: bool = False,
) -> list[dict]:
    """
    Stream WildChat-1M and collect benign multi-turn conversations that have
    very low self-harm / self-harm-intent moderation scores.

    Returns a list of record dicts ready to be merged into the psychotherapy
    DataFrame (keys: text, conversations, label, original_index, pairing_id,
    session_id, num_turns, num_sessions_context, source, moderation_scores).
    """
    print(f"\n{'='*70}")
    print("  AUGMENTING NEGATIVES FROM WildChat-1M")
    print(f"{'='*70}")
    print(f"  Target negatives    : {n_needed}")
    print(f"  Self-harm threshold : {config.PSYCHOTHERAPY_AUGMENT_SELFHARM_THRESHOLD}")
    print(f"  Violence threshold  : {config.PSYCHOTHERAPY_AUGMENT_VIOLENCE_THRESHOLD}")
    print(f"  Min turns           : {config.PSYCHOTHERAPY_AUGMENT_MIN_TURNS}")

    ds = load_dataset(
        config.PSYCHOTHERAPY_AUGMENT_SOURCE, split="train", streaming=True,
    )

    tok = _load_truncation_tokenizer()
    over_collect = int(n_needed * 3)  # collect extra, sample later
    max_scan = 5_000 if dry_run else None

    records: list[dict] = []
    stats = {
        "total_scanned": 0,
        "skipped_lang": 0,
        "skipped_no_mod": 0,
        "skipped_selfharm": 0,
        "skipped_violence": 0,
        "skipped_few_turns": 0,
    }

    sh_thresh = config.PSYCHOTHERAPY_AUGMENT_SELFHARM_THRESHOLD
    shi_thresh = config.PSYCHOTHERAPY_AUGMENT_SELFHARM_INTENT_THRESHOLD
    v_thresh = config.PSYCHOTHERAPY_AUGMENT_VIOLENCE_THRESHOLD
    h_thresh = config.PSYCHOTHERAPY_AUGMENT_HARASSMENT_THRESHOLD
    min_turns = config.PSYCHOTHERAPY_AUGMENT_MIN_TURNS

    for row in tqdm(ds, desc="Scanning WildChat-1M", total=max_scan):
        stats["total_scanned"] += 1
        if max_scan and stats["total_scanned"] > max_scan:
            break

        # English only
        if (row.get("language") or "").lower() not in ("english", "en"):
            stats["skipped_lang"] += 1
            continue

        # Must have moderation scores
        oai_mod = row.get("openai_moderation")
        if not oai_mod:
            stats["skipped_no_mod"] += 1
            continue

        # Filter: low self-harm across ALL turns
        sh = _wildchat_max_category_score(oai_mod, "self-harm")
        shi = _wildchat_max_category_score(oai_mod, "self-harm/intent")
        if sh >= sh_thresh or shi >= shi_thresh:
            stats["skipped_selfharm"] += 1
            continue

        # Also reject high violence / harassment
        v = _wildchat_max_category_score(oai_mod, "violence")
        har = _wildchat_max_category_score(oai_mod, "harassment")
        if v >= v_thresh or har >= h_thresh:
            stats["skipped_violence"] += 1
            continue

        # Extract turns
        conversation = row.get("conversation", [])
        turns = []
        for turn in conversation:
            role = turn.get("role", "unknown")
            content = (turn.get("content") or "").strip()
            if content and role in ("user", "assistant"):
                turns.append({"role": role, "content": content})

        if len(turns) < min_turns:
            stats["skipped_few_turns"] += 1
            continue

        text = _format_turns_as_text(turns)
        token_count = _count_tokens_exact(text, tok)

        records.append({
            "text": text,
            "conversations": json.dumps(turns),
            "label": 0,
            "original_index": 1_000_000 + stats["total_scanned"],
            "pairing_id": -1,               # sentinel: not from psychotherapy
            "session_id": -1,
            "num_turns": len(turns),
            "num_sessions_context": 0,
            "source": "wildchat",
            "token_count": token_count,
            "moderation_scores": json.dumps({
                "source": "wildchat",
                "max_self_harm": round(sh, 6),
                "max_self_harm_intent": round(shi, 6),
            }),
        })

        if len(records) >= over_collect:
            print(f"\n  Over-collected {len(records)} after scanning "
                  f"{stats['total_scanned']:,} rows")
            break

    print(f"\n  WildChat scan summary:")
    for k, v in stats.items():
        print(f"    {k:25s}: {v:,}")
    print(f"    kept_benign           : {len(records):,}")

    if len(records) < n_needed:
        print(f"\n  ⚠ Only collected {len(records)} negatives "
              f"(target was {n_needed}). Using all of them.")
        return records

    # Sub-sample to exactly n_needed
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(records), size=n_needed, replace=False)
    sampled = [records[i] for i in idx]
    print(f"  Sub-sampled {n_needed} negatives from {len(records)} candidates")
    return sampled


def prepare_psychotherapy(args: argparse.Namespace) -> None:
    """
    Psychotherapy adverse-event classification pipeline.

    Downloads the AI Psychotherapy Eval dataset (Steenstra, 2025) and builds
    a binary classification task: predict whether a therapy session results
    in any adverse event (suicide, self-harm, relapse, dropout, etc.).

    Two modes controlled by ``args.dataset``:

    * **psychotherapy-single** — each ``(pairing_id, session_id)`` is one
      independent example.  The conversation is the turns from that single
      session only.

    * **psychotherapy-sliding** — for session *N* of a pairing, the input
      is the concatenation of sessions 1 … N (with ``[Session K]`` markers).
      The label is whether session *N* had an adverse event.  This tests
      whether earlier context helps predict later adverse events.
    """
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sliding = args.dataset == "psychotherapy-sliding"
    mode_label = "SLIDING-WINDOW" if sliding else "SINGLE-SESSION"

    print(f"\n{'='*70}")
    print(f"  PSYCHOTHERAPY ADVERSE-EVENT CLASSIFIER — {mode_label}")
    print(f"{'='*70}")
    print(f"  Conversation source : {config.PSYCHOTHERAPY_CONVERSATION_URL.split('/')[-1]}")
    print(f"  Adverse events      : {config.PSYCHOTHERAPY_ADVERSE_EVENTS_URL.split('/')[-1]}")
    print(f"  Max seq len         : {config.PSYCHOTHERAPY_MAX_SEQ_LEN}")

    # ----- Download data ----- #
    cache_dir = output_dir / "raw_cache"
    conv_df = _download_csv(
        config.PSYCHOTHERAPY_CONVERSATION_URL,
        cache_dir / "conversation_log.csv",
    )
    adverse_df = _download_csv(
        config.PSYCHOTHERAPY_ADVERSE_EVENTS_URL,
        cache_dir / "adverse_events.csv",
    )

    print(f"\n  Conversation rows   : {len(conv_df):,}")
    print(f"  Adverse event rows  : {len(adverse_df):,}")

    # ----- Build labels ----- #
    session_labels = _build_session_labels(adverse_df)
    print(f"  Unique sessions     : {len(session_labels):,}")
    n_pos = int(session_labels["label"].sum())
    n_neg = len(session_labels) - n_pos
    print(f"  Sessions with adverse events    : {n_pos}")
    print(f"  Sessions without adverse events : {n_neg}")

    # Per-event-type breakdown
    adverse_occurred = adverse_df[adverse_df["occurred"].astype(bool)]
    event_counts = adverse_occurred["event_type"].value_counts()
    print(f"\n  Per-event-type breakdown (across all sessions):")
    for etype, cnt in event_counts.items():
        print(f"    {etype:45s}: {cnt}")

    # ----- Build conversations ----- #
    conversations = _build_conversations(conv_df)
    print(f"\n  Sessions with dialogue : {len(conversations)}")

    # ----- Assemble examples ----- #
    # Get all unique pairings and their session counts
    all_sessions = sorted(session_labels[["pairing_id", "session_id"]].itertuples(index=False))

    # Group sessions by pairing for sliding-window
    from collections import defaultdict
    sessions_by_pairing: dict[int, list[int]] = defaultdict(list)
    for pid, sid in all_sessions:
        sessions_by_pairing[pid].append(sid)
    for pid in sessions_by_pairing:
        sessions_by_pairing[pid].sort()

    records: list[dict] = []
    skipped_no_conv = 0
    idx = 0

    for pid, sid in all_sessions:
        # Get label for this specific session
        label_row = session_labels[
            (session_labels["pairing_id"] == pid) &
            (session_labels["session_id"] == sid)
        ]
        if label_row.empty:
            continue
        label = int(label_row["label"].iloc[0])

        if sliding:
            # Sliding window: concatenate sessions 1..sid for this pairing
            prior_sessions = [
                s for s in sessions_by_pairing[pid] if s <= sid
            ]
            all_turns = []
            for s in prior_sessions:
                key = (pid, s)
                if key not in conversations:
                    continue
                # Add session marker
                all_turns.append({
                    "role": "system",
                    "content": f"[Session {s}]",
                })
                all_turns.extend(conversations[key])

            if not all_turns:
                skipped_no_conv += 1
                continue
        else:
            # Single session: just this session's turns
            key = (pid, sid)
            if key not in conversations:
                skipped_no_conv += 1
                continue
            all_turns = conversations[key]

        text = _format_turns_as_text(all_turns)
        records.append({
            "text": text,
            "conversations": json.dumps(all_turns),
            "label": label,
            "original_index": idx,
            "pairing_id": pid,
            "session_id": sid,
            "num_turns": len(all_turns),
            "num_sessions_context": len([s for s in sessions_by_pairing[pid] if s <= sid]) if sliding else 1,
            "moderation_scores": json.dumps({
                "source": "psychotherapy",
                "mode": "sliding" if sliding else "single",
            }),
        })
        idx += 1

    print(f"\n  Total examples         : {len(records)}")
    print(f"  Skipped (no dialogue)  : {skipped_no_conv}")

    if len(records) == 0:
        print("\n  ✗ No examples produced. Check the CSV downloads. Exiting.")
        sys.exit(1)

    df = pd.DataFrame(records)
    df["source"] = "psychotherapy"

    # Class distribution (before augmentation)
    n_pos = int(df["label"].sum())
    n_neg = len(df) - n_pos
    print(f"  Positive (adverse)     : {n_pos} ({100*n_pos/len(df):.1f}%)")
    print(f"  Negative (no adverse)  : {n_neg} ({100*n_neg/len(df):.1f}%)")

    # ----- WildChat negative augmentation ----- #
    augment = getattr(args, "augment_negatives", False)
    if augment:
        # Target 50/50: need (n_pos - n_neg) more negatives
        n_needed = max(0, n_pos - n_neg)
        if n_needed > 0:
            wc_records = _sample_wildchat_negatives(
                n_needed, seed=args.seed, dry_run=args.dry_run,
            )
            if wc_records:
                wc_df = pd.DataFrame(wc_records)
                # Ensure columns align
                for col in df.columns:
                    if col not in wc_df.columns:
                        wc_df[col] = None
                df = pd.concat([df, wc_df[df.columns]], ignore_index=True)
                n_pos_new = int(df["label"].sum())
                n_neg_new = len(df) - n_pos_new
                print(f"\n  After augmentation:")
                print(f"    Total examples     : {len(df)}")
                print(f"    Positive (adverse) : {n_pos_new} ({100*n_pos_new/len(df):.1f}%)")
                print(f"    Negative (benign)  : {n_neg_new} ({100*n_neg_new/len(df):.1f}%)")
                print(f"    From psychotherapy : {len(df[df['source'] == 'psychotherapy'])}")
                print(f"    From WildChat      : {len(df[df['source'] == 'wildchat'])}")
        else:
            print(f"\n  No augmentation needed (already balanced: {n_pos} pos / {n_neg} neg)")

    if sliding:
        print(f"\n  Sessions-in-context stats:")
        print(f"    mean: {df['num_sessions_context'].mean():.1f}, "
              f"max: {df['num_sessions_context'].max()}")

    # ----- Token count stats ----- #
    tok = _load_truncation_tokenizer()
    df["token_count"] = df["text"].apply(lambda t: _count_tokens_exact(t, tok))
    print(f"\n  Token counts — mean: {df['token_count'].mean():.0f}, "
          f"median: {df['token_count'].median():.0f}, "
          f"min: {df['token_count'].min()}, max: {df['token_count'].max()}")
    print(f"  Turn counts  — mean: {df['num_turns'].mean():.1f}, "
          f"median: {df['num_turns'].median():.0f}")

    # ----- Split ----- #
    if augment and len(df) > config.PSYCHOTHERAPY_TOTAL_SIZE:
        split_sizes = config.PSYCHOTHERAPY_AUGMENTED_SPLIT_SIZES
    else:
        split_sizes = config.PSYCHOTHERAPY_SPLIT_SIZES
    df = create_splits(df, args.seed, args.dry_run, split_sizes_override=split_sizes)
    df = _maybe_insert_canaries(df, args, output_dir)

    # ----- Save ----- #
    output_file = output_dir / config.DATA_FILE
    df.to_parquet(output_file, index=False)
    print(f"\n  Saved to {output_file}")
    save_dataset_metadata(
        output_dir,
        args.dataset,
        max_seq_len=config.PSYCHOTHERAPY_MAX_SEQ_LEN,
        mode="sliding" if sliding else "single",
        adverse_event_types=config.ADVERSE_EVENT_TYPES,
        augmented_negatives=augment,
        augment_source=config.PSYCHOTHERAPY_AUGMENT_SOURCE if augment else None,
    )
    print_summary(df, output_file)


# =========================================================================== #
# Emotional-support pipeline (merged psychotherapy + ESConv + WildChat)
# =========================================================================== #

def _load_esconv_positives(dry_run: bool = False) -> tuple[list[dict], dict]:
    """Load ESConv conversations as positive (support-needed) examples.

    Each row in ``thu-coai/esconv`` has a ``text`` field containing a JSON
    string with keys ``dialog``, ``emotion_type``, ``problem_type``, and
    ``survey_score``.  Speaker roles ``"usr"``/``"sys"`` are mapped to
    ``"user"``/``"assistant"``.

    Returns
    -------
    records : list[dict]
        Records ready for the pipeline DataFrame.
    stats : dict
        Loading statistics (total, skipped, kept).
    """
    print(f"\n{'='*70}")
    print("  LOADING ESConv (thu-coai/esconv)")
    print(f"{'='*70}")

    ds = load_dataset(config.ESCONV_DATASET_ID, split="train")
    print(f"  Rows in train split: {len(ds):,}")

    if dry_run:
        ds = ds.select(range(min(100, len(ds))))
        print(f"  Dry-run: using first {len(ds)} rows")

    ROLE_MAP = {"usr": "user", "sys": "assistant"}
    records: list[dict] = []
    stats = {"total": len(ds), "skipped_parse": 0, "skipped_empty": 0, "kept": 0}

    for idx, row in enumerate(tqdm(ds, desc="Parsing ESConv")):
        raw_text = row.get("text", "")
        try:
            data = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            stats["skipped_parse"] += 1
            continue

        dialog = data.get("dialog", [])
        if not dialog:
            stats["skipped_empty"] += 1
            continue

        # Extract turns
        turns = []
        for turn in dialog:
            speaker = turn.get("speaker", "")
            role = ROLE_MAP.get(speaker)
            if role is None:
                continue
            content = (turn.get("text") or turn.get("content") or "").strip()
            if content:
                turns.append({"role": role, "content": content})

        if len(turns) < 2:
            stats["skipped_empty"] += 1
            continue

        # Extract metadata for severity analysis
        emotion_type = data.get("emotion_type", "unknown")
        problem_type = data.get("problem_type", "unknown")
        survey = data.get("survey_score", {})
        seeker_survey = survey.get("seeker", {}) if isinstance(survey, dict) else {}
        emotion_intensity = seeker_survey.get("initial_emotion_intensity", -1)
        try:
            emotion_intensity = int(emotion_intensity)
        except (ValueError, TypeError):
            emotion_intensity = -1

        text = _format_turns_as_text(turns)
        records.append({
            "text": text,
            "conversations": json.dumps(turns),
            "label": 1,  # all ESConv = support-needed
            "original_index": 2_000_000 + idx,
            "pairing_id": -2,           # sentinel: from ESConv
            "session_id": -1,
            "num_turns": len(turns),
            "num_sessions_context": 1,
            "source": "esconv",
            "emotion_type": emotion_type,
            "problem_type": problem_type,
            "emotion_intensity": emotion_intensity,
            "moderation_scores": json.dumps({
                "source": "esconv",
                "emotion_type": emotion_type,
                "problem_type": problem_type,
                "emotion_intensity": emotion_intensity,
            }),
        })
        stats["kept"] += 1

    print(f"\n  ESConv loading summary:")
    for k, v in stats.items():
        print(f"    {k:25s}: {v:,}")

    return records, stats


def prepare_emotional_support(args: argparse.Namespace) -> None:
    """
    Emotional-support binary classification pipeline.

    Combines two positive sources (all conversations = "support needed"):
      1. Psychotherapy (Steenstra 2025) — all sessions, regardless of
         adverse event label, treated as support-needed.
      2. ESConv (thu-coai/esconv) — emotional support conversations.

    Negatives come from WildChat-1M (benign conversations with low
    self-harm/violence scores).

    Target: ~3,600 examples at 50/50 balance.
    """
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("  EMOTIONAL-SUPPORT CLASSIFIER — MERGED DATASET")
    print(f"{'='*70}")

    # ---- Source 1: Psychotherapy (all sessions → positive) ---- #
    print(f"\n--- Loading Psychotherapy (all sessions → support-needed) ---")
    cache_dir = output_dir / "raw_cache"
    conv_df = _download_csv(
        config.PSYCHOTHERAPY_CONVERSATION_URL,
        cache_dir / "conversation_log.csv",
    )
    adverse_df = _download_csv(
        config.PSYCHOTHERAPY_ADVERSE_EVENTS_URL,
        cache_dir / "adverse_events.csv",
    )

    # Build conversations (reuse existing helper)
    conversations = _build_conversations(conv_df)

    # We still need session labels to know which (pairing, session) pairs exist
    session_labels = _build_session_labels(adverse_df)

    # Get unique sessions
    all_sessions = sorted(
        session_labels[["pairing_id", "session_id"]].itertuples(index=False)
    )

    psych_records: list[dict] = []
    idx = 0
    skipped = 0
    for pid, sid in all_sessions:
        key = (int(pid), int(sid))
        if key not in conversations:
            skipped += 1
            continue
        turns = conversations[key]
        text = _format_turns_as_text(turns)

        # Original adverse-event label (kept as metadata, NOT used for training)
        label_row = session_labels[
            (session_labels["pairing_id"] == pid) &
            (session_labels["session_id"] == sid)
        ]
        original_adverse = int(label_row["label"].iloc[0]) if not label_row.empty else 0

        psych_records.append({
            "text": text,
            "conversations": json.dumps(turns),
            "label": 1,  # ALL psychotherapy = support-needed
            "original_index": idx,
            "pairing_id": int(pid),
            "session_id": int(sid),
            "num_turns": len(turns),
            "num_sessions_context": 1,
            "source": "psychotherapy",
            "emotion_type": "unknown",
            "problem_type": "therapy_session",
            "emotion_intensity": -1,
            "original_adverse_label": original_adverse,
            "moderation_scores": json.dumps({
                "source": "psychotherapy",
                "original_adverse_label": original_adverse,
            }),
        })
        idx += 1

    print(f"  Psychotherapy sessions: {len(psych_records)} "
          f"(skipped {skipped} without dialogue)")

    # ---- Source 2: ESConv ---- #
    esconv_records, esconv_stats = _load_esconv_positives(dry_run=args.dry_run)

    # ---- Combine positives ---- #
    all_positives = psych_records + esconv_records
    n_positives = len(all_positives)
    print(f"\n  Total positives: {n_positives}")
    print(f"    From psychotherapy: {len(psych_records)}")
    print(f"    From ESConv:        {len(esconv_records)}")

    # ---- Source 3: WildChat negatives (match count for 50/50) ---- #
    n_negatives_needed = n_positives
    wc_records = _sample_wildchat_negatives(
        n_negatives_needed, seed=args.seed, dry_run=args.dry_run,
    )
    # Add missing metadata columns to WildChat records
    for rec in wc_records:
        rec.setdefault("emotion_type", "none")
        rec.setdefault("problem_type", "none")
        rec.setdefault("emotion_intensity", -1)
        rec.setdefault("original_adverse_label", -1)

    print(f"  WildChat negatives:   {len(wc_records)}")

    # ---- Assemble full DataFrame ---- #
    all_records = all_positives + wc_records
    df = pd.DataFrame(all_records)

    n_total = len(df)
    n_pos = int(df["label"].sum())
    n_neg = n_total - n_pos
    print(f"\n  Final dataset:")
    print(f"    Total examples     : {n_total}")
    print(f"    Positive (support) : {n_pos} ({100*n_pos/n_total:.1f}%)")
    print(f"    Negative (benign)  : {n_neg} ({100*n_neg/n_total:.1f}%)")

    by_source = df.groupby("source").size()
    for src, cnt in by_source.items():
        print(f"    Source '{src}': {cnt}")

    # ---- ESConv severity breakdown ---- #
    esconv_mask = df["source"] == "esconv"
    if esconv_mask.any():
        print(f"\n  ESConv emotion intensity distribution:")
        intensity_dist = df.loc[esconv_mask, "emotion_intensity"].value_counts().sort_index()
        for val, cnt in intensity_dist.items():
            print(f"    intensity={val}: {cnt}")

        print(f"\n  ESConv problem type distribution:")
        problem_dist = df.loc[esconv_mask, "problem_type"].value_counts()
        for ptype, cnt in problem_dist.head(10).items():
            print(f"    {ptype}: {cnt}")

    # ---- Token count stats ---- #
    tok = _load_truncation_tokenizer()
    df["token_count"] = df["text"].apply(lambda t: _count_tokens_exact(t, tok))
    print(f"\n  Token counts — mean: {df['token_count'].mean():.0f}, "
          f"median: {df['token_count'].median():.0f}, "
          f"min: {df['token_count'].min()}, max: {df['token_count'].max()}")

    # ---- Split ---- #
    split_sizes = config.EMOTIONAL_SUPPORT_SPLIT_SIZES
    # Adjust if we have fewer examples than expected
    total_available = len(df)
    total_target = config.EMOTIONAL_SUPPORT_TOTAL_SIZE
    if total_available < total_target:
        ratio = total_available / total_target
        split_sizes = {k: max(10, int(v * ratio)) for k, v in split_sizes.items()}
        print(f"\n  ⚠ Adjusted split sizes (only {total_available} examples available):")
        for k, v in split_sizes.items():
            print(f"    {k}: {v}")

    df = create_splits(df, args.seed, args.dry_run, split_sizes_override=split_sizes)
    df = _maybe_insert_canaries(df, args, output_dir)

    # ---- Save ---- #
    output_file = output_dir / config.DATA_FILE
    df.to_parquet(output_file, index=False)
    print(f"\n  Saved to {output_file}")
    save_dataset_metadata(
        output_dir,
        args.dataset,
        max_seq_len=config.EMOTIONAL_SUPPORT_MAX_SEQ_LEN,
        mode="emotional-support",
        sources=["psychotherapy", "esconv", "wildchat"],
        n_psychotherapy=len(psych_records),
        n_esconv=len(esconv_records),
        n_wildchat=len(wc_records),
        esconv_stats=esconv_stats,
    )
    print_summary(df, output_file)


# =========================================================================== #
# Pooled multi-source pipeline
# =========================================================================== #

def prepare_pooled(args: argparse.Namespace) -> None:
    """
    Pooled binary safety classifier: merge BeaverTails (single-turn),
    XGuard + WildChat (multi-turn), and emotional-support (multi-session)
    into a single dataset with a unified safe/unsafe label.

    All records include a ``conversations`` column (JSON-encoded turn list)
    so that the entire dataset flows through ``format_multiturn_for_classification``
    at training and inference time.  Single-turn BeaverTails prompts are
    wrapped as ``[{"role": "user", "content": text}]``.
    """
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_source_target = config.POOLED_PER_SOURCE_TARGET
    tok = _load_truncation_tokenizer()

    print(f"\n{'='*70}")
    print("  POOLED MULTI-SOURCE SAFETY CLASSIFIER")
    print(f"{'='*70}")
    print(f"  Sources: BeaverTails, XGuard multi-turn, emotional-support")
    print(f"  Target per source: {per_source_target}")
    print(f"  Max seq len: {config.POOLED_MAX_SEQ_LEN}")

    # ================================================================== #
    #  Source 1: BeaverTails (single-turn QA pairs)
    # ================================================================== #
    print(f"\n--- Source 1: BeaverTails (single-turn) ---")
    bt_dataset_id = config.DATASET_IDS["beavertails"]
    from datasets import concatenate_datasets

    ds_bt_train = load_dataset(bt_dataset_id, split="330k_train")
    ds_bt_test = load_dataset(bt_dataset_id, split="330k_test")
    ds_bt = concatenate_datasets([ds_bt_train, ds_bt_test])
    print(f"  Total rows loaded: {len(ds_bt):,}")

    if args.dry_run:
        ds_bt = ds_bt.select(range(min(config.DRY_RUN_SIZE, len(ds_bt))))

    bt_pos: list[dict] = []
    bt_neg: list[dict] = []
    for i, row in enumerate(tqdm(ds_bt, desc="Processing BeaverTails")):
        text = (row.get("prompt") or "").strip()
        if not text:
            continue
        is_safe = row.get("is_safe", True)
        label = 0 if is_safe else 1

        record = {
            "text": text,
            "conversations": json.dumps([{"role": "user", "content": text}]),
            "label": label,
            "original_index": i,
            "source": "beavertails",
            "format_type": "single_turn",
            "moderation_scores": json.dumps({"source": "beavertails", "is_safe": bool(is_safe)}),
        }
        (bt_neg if label == 0 else bt_pos).append(record)

    print(f"  BeaverTails: {len(bt_pos):,} unsafe, {len(bt_neg):,} safe")

    # ================================================================== #
    #  Source 2: XGuard multi-turn (harmful) + WildChat benign (multi-turn)
    # ================================================================== #
    print(f"\n--- Source 2: XGuard + WildChat multi-turn ---")
    xg_dataset_id = config.DATASET_IDS["xguard-multiturn"]
    xguard_ds = load_dataset(xg_dataset_id, split="train")
    print(f"  XGuard conversations loaded: {len(xguard_ds):,}")

    if args.dry_run:
        xguard_ds = xguard_ds.select(range(min(500, len(xguard_ds))))

    xg_pos: list[dict] = []
    for i, row in enumerate(tqdm(xguard_ds, desc="Processing XGuard")):
        conversations_raw = row.get("conversations", [])
        if not conversations_raw:
            continue
        turns = []
        for turn in conversations_raw:
            role = _xguard_role(turn.get("from", ""))
            content = (turn.get("value") or "").strip()
            if content:
                turns.append({"role": role, "content": content})
        if len(turns) < 2:
            continue

        text = _format_turns_as_text(turns)
        xg_pos.append({
            "text": text,
            "conversations": json.dumps(turns),
            "label": 1,
            "original_index": 1_000_000 + i,
            "source": "xguard",
            "format_type": "multi_turn",
            "moderation_scores": json.dumps({"source": "xguard"}),
        })

    print(f"  XGuard harmful: {len(xg_pos):,}")

    # WildChat benign multi-turn
    wildchat_ds = load_dataset(
        config.XGUARD_BENIGN_DATASET_ID, split="train", streaming=True,
    )
    xg_neg: list[dict] = []
    max_benign = len(xg_pos) * 3
    max_scan = 2_000 if args.dry_run else None
    wc_seen = 0
    for row in tqdm(wildchat_ds, desc="Scanning WildChat multi-turn", total=max_scan):
        wc_seen += 1
        if max_scan and wc_seen > max_scan:
            break
        if (row.get("language") or "").lower() not in ("english", "en"):
            continue
        conversation = row.get("conversation", [])
        if len(conversation) < 3:
            continue
        turns = []
        for turn in conversation:
            role = turn.get("role", "unknown")
            content = (turn.get("content") or "").strip()
            if content and role in ("user", "assistant"):
                turns.append({"role": role, "content": content})
        if len(turns) < 3:
            continue

        text = _format_turns_as_text(turns)
        xg_neg.append({
            "text": text,
            "conversations": json.dumps(turns),
            "label": 0,
            "original_index": 2_000_000 + wc_seen,
            "source": "xguard",
            "format_type": "multi_turn",
            "moderation_scores": json.dumps({"source": "wildchat"}),
        })
        if len(xg_neg) >= max_benign:
            break

    print(f"  WildChat benign multi-turn: {len(xg_neg):,}")

    # ================================================================== #
    #  Source 3: Emotional-support (psychotherapy + ESConv + WildChat neg)
    # ================================================================== #
    print(f"\n--- Source 3: Emotional-support (multi-session) ---")
    cache_dir = output_dir / "raw_cache"

    # Psychotherapy positives
    conv_df = _download_csv(
        config.PSYCHOTHERAPY_CONVERSATION_URL,
        cache_dir / "conversation_log.csv",
    )
    adverse_df = _download_csv(
        config.PSYCHOTHERAPY_ADVERSE_EVENTS_URL,
        cache_dir / "adverse_events.csv",
    )
    conversations = _build_conversations(conv_df)
    session_labels = _build_session_labels(adverse_df)
    all_sessions = sorted(
        session_labels[["pairing_id", "session_id"]].itertuples(index=False)
    )

    es_pos: list[dict] = []
    idx = 0
    for pid, sid in all_sessions:
        key = (int(pid), int(sid))
        if key not in conversations:
            continue
        turns = conversations[key]
        text = _format_turns_as_text(turns)
        es_pos.append({
            "text": text,
            "conversations": json.dumps(turns),
            "label": 1,
            "original_index": 3_000_000 + idx,
            "source": "emotional-support",
            "format_type": "multi_session",
            "moderation_scores": json.dumps({"source": "psychotherapy"}),
        })
        idx += 1

    print(f"  Psychotherapy positives: {len(es_pos)}")

    # ESConv positives
    esconv_records, _ = _load_esconv_positives(dry_run=args.dry_run)
    for rec in esconv_records:
        es_pos.append({
            "text": rec["text"],
            "conversations": rec.get("conversations", json.dumps(
                [{"role": "user", "content": rec["text"]}]
            )),
            "label": 1,
            "original_index": 4_000_000 + len(es_pos),
            "source": "emotional-support",
            "format_type": "multi_session",
            "moderation_scores": json.dumps({"source": "esconv"}),
        })

    print(f"  Total emotional-support positives: {len(es_pos)}")

    # WildChat negatives for emotional-support
    n_es_neg = len(es_pos)
    wc_neg_raw = _sample_wildchat_negatives(n_es_neg, seed=args.seed, dry_run=args.dry_run)
    es_neg: list[dict] = []
    for rec in wc_neg_raw:
        es_neg.append({
            "text": rec["text"],
            "conversations": rec.get("conversations", json.dumps(
                [{"role": "user", "content": rec["text"]}]
            )),
            "label": 0,
            "original_index": 5_000_000 + len(es_neg),
            "source": "emotional-support",
            "format_type": "multi_session",
            "moderation_scores": rec.get("moderation_scores", json.dumps({"source": "wildchat"})),
        })

    print(f"  Emotional-support negatives: {len(es_neg)}")

    # ================================================================== #
    #  Stratified sampling: balance across sources, 50/50 per source
    # ================================================================== #
    print(f"\n--- Stratified sampling ---")

    rng = np.random.RandomState(args.seed)
    source_pools = {
        "beavertails": (bt_pos, bt_neg),
        "xguard": (xg_pos, xg_neg),
        "emotional-support": (es_pos, es_neg),
    }

    # Determine per-source allocations: emotional-support may be smaller
    source_allocations: dict[str, int] = {}
    shortfall = 0
    expandable_sources = []
    for src_name, (pos, neg) in source_pools.items():
        available = min(len(pos), len(neg)) * 2  # max balanced total
        if args.dry_run:
            available = min(available, 200)
        target = per_source_target if not args.dry_run else min(200, per_source_target)
        if available < target:
            source_allocations[src_name] = available
            shortfall += target - available
        else:
            source_allocations[src_name] = target
            expandable_sources.append(src_name)

    # Redistribute shortfall to sources with headroom
    if shortfall > 0 and expandable_sources:
        extra_per = shortfall // len(expandable_sources)
        remainder = shortfall % len(expandable_sources)
        for i, src_name in enumerate(expandable_sources):
            pos, neg = source_pools[src_name]
            max_avail = min(len(pos), len(neg)) * 2
            bonus = extra_per + (1 if i < remainder else 0)
            source_allocations[src_name] = min(
                source_allocations[src_name] + bonus, max_avail,
            )

    for src_name, alloc in source_allocations.items():
        print(f"  {src_name}: {alloc} examples allocated")

    # Sample balanced subsets
    all_records: list[dict] = []
    for src_name, (pos, neg) in source_pools.items():
        n_total = source_allocations[src_name]
        n_pos = n_total // 2
        n_neg = n_total - n_pos
        n_pos = min(n_pos, len(pos))
        n_neg = min(n_neg, len(neg))

        pos_idx = rng.choice(len(pos), size=n_pos, replace=False)
        neg_idx = rng.choice(len(neg), size=n_neg, replace=False)
        all_records.extend(pos[i] for i in pos_idx)
        all_records.extend(neg[i] for i in neg_idx)

    df = pd.DataFrame(all_records)
    n_total = len(df)
    n_pos = int(df["label"].sum())
    n_neg = n_total - n_pos
    print(f"\n  Pooled dataset: {n_total} total ({n_pos} unsafe, {n_neg} safe)")

    by_source = df.groupby("source").size()
    for src, cnt in by_source.items():
        print(f"    {src}: {cnt}")

    # ---- Token counts ---- #
    print("  Counting tokens…")
    df["token_count"] = df["text"].apply(lambda t: _count_tokens_exact(t, tok))
    print(f"  Token counts — mean: {df['token_count'].mean():.0f}, "
          f"median: {df['token_count'].median():.0f}, "
          f"min: {df['token_count'].min()}, max: {df['token_count'].max()}")

    # ---- Truncate to POOLED_MAX_SEQ_LEN ---- #
    orig_max_seq = config.MAX_SEQ_LEN
    config.MAX_SEQ_LEN = config.POOLED_MAX_SEQ_LEN
    df = truncate_texts(df)
    config.MAX_SEQ_LEN = orig_max_seq

    # Re-count after truncation
    df["token_count"] = df["text"].apply(lambda t: _count_tokens_exact(t, tok))

    # ---- Split ---- #
    split_sizes = config.POOLED_SPLIT_SIZES
    if args.dry_run:
        split_sizes = {k: min(40, len(df) // 5) for k in split_sizes}
    df = create_splits(df, args.seed, args.dry_run, split_sizes_override=split_sizes)
    df = _maybe_insert_canaries(df, args, output_dir)

    # ---- Save ---- #
    output_file = output_dir / config.DATA_FILE
    df.to_parquet(output_file, index=False)
    print(f"\n  Saved to {output_file}")

    source_counts = df.groupby("source").size().to_dict()
    save_dataset_metadata(
        output_dir,
        "pooled",
        max_seq_len=config.POOLED_MAX_SEQ_LEN,
        sources=list(source_counts.keys()),
        source_counts=source_counts,
        format_types=df["format_type"].unique().tolist(),
    )
    print_summary(df, output_file)


# =========================================================================== #
# CLI
# =========================================================================== #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1: Prepare labeled data for membership inference experiments.",
    )
    p.add_argument(
        "--dataset",
        choices=list(config.DATASET_IDS.keys()),
        default=config.DEFAULT_DATASET_VARIANT,
        help=(
            "Which dataset variant to use (default: %(default)s). "
            "'xguard-multiturn' runs the long-context multi-turn experiment."
        ),
    )
    p.add_argument(
        "--categories",
        type=str,
        default=",".join(config.DEFAULT_MODERATION_THRESHOLDS.keys()),
        help="Comma-separated moderation categories (WildChat only).",
    )
    p.add_argument(
        "--thresholds",
        type=str,
        default=",".join(str(v) for v in config.DEFAULT_MODERATION_THRESHOLDS.values()),
        help="Comma-separated score thresholds (WildChat only).",
    )
    p.add_argument("--total_size", type=int, default=config.TOTAL_SIZE)
    p.add_argument("--pos_neg_ratio", type=float, default=config.POS_NEG_RATIO)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--output_dir", type=str, default=None,
                   help="Override output directory (default: config.DATA_DIR, "
                        "or <run_dir>/data when --run_dir is set).")
    p.add_argument("--run_dir", type=str, default=None,
                   help="Timestamped run directory (e.g. results/2026-02-14_153000). "
                        "When set, data is written to <run_dir>/data/.")
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Process only a small sample for pipeline testing.",
    )
    p.add_argument(
        "--bt_categories",
        type=str,
        default="",
        help=(
            "Comma-separated BeaverTails harm categories to filter on. "
            "When set, only rows matching at least one selected category "
            "are labelled positive. Empty string (default) uses the "
            "overall is_safe flag.  Valid categories: "
            + ", ".join(config.BEAVERTAILS_CATEGORIES)
        ),
    )
    p.add_argument(
        "--label_mode",
        choices=["binary", "multiclass"],
        default="binary",
        help=(
            "Label mode for BeaverTails. 'binary' (default) uses safe/unsafe. "
            "'multiclass' assigns each example to one of 15 classes "
            "(14 harm categories + safe)."
        ),
    )
    # ---- Canary experiment flags ----
    p.add_argument(
        "--canary",
        action="store_true",
        help="Enable synthetic canary insertion for spurious label memorization.",
    )
    p.add_argument(
        "--canary_fraction",
        type=float,
        default=config.CANARY_FRACTION,
        help="Fraction of each training split to receive canaries (default: %(default)s).",
    )
    p.add_argument(
        "--canary_repeats",
        type=int,
        default=config.CANARY_REPEATS,
        help="Number of training examples sharing one canary (default: %(default)s).",
    )
    p.add_argument(
        "--canary_position",
        type=str,
        choices=list(config.CANARY_POSITIONS),
        default=config.CANARY_POSITION,
        help="Where to insert canary text in each document: start, middle, or end (default: %(default)s).",
    )
    # ---- Psychotherapy augmentation flags ----
    p.add_argument(
        "--augment_negatives",
        action="store_true",
        help=(
            "Augment with benign WildChat-1M conversations to balance "
            "the psychotherapy dataset (50/50 pos/neg). Only applies "
            "to psychotherapy-single and psychotherapy-sliding."
        ),
    )
    return p.parse_args()


def _maybe_insert_canaries(
    df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    """Insert canaries if ``--canary`` flag is set, save metadata, return df."""
    if not getattr(args, "canary", False):
        return df

    # Determine num_labels so canaries span all categories in multiclass
    label_mode = getattr(args, "label_mode", "binary")
    dataset = getattr(args, "dataset", None)
    num_labels = config.get_num_labels(dataset, label_mode)

    canary_position = getattr(args, "canary_position", config.CANARY_POSITION)
    df, canary_meta = insert_canaries(
        df,
        canary_fraction=args.canary_fraction,
        canary_repeats=args.canary_repeats,
        num_labels=num_labels,
        position=canary_position,
    )
    # Save canary metadata alongside the parquet
    meta_path = output_dir / "canary_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(canary_meta, f, indent=2)
    print(f"  Canary metadata saved to {meta_path}")

    return df


if __name__ == "__main__":
    args = parse_args()

    # Scope paths into the run directory when --run_dir is provided
    if args.run_dir:
        config.set_run_dir(args.run_dir)

    # Resolve output_dir: explicit flag > run-scoped DATA_DIR > default DATA_DIR
    if args.output_dir is None:
        args.output_dir = str(config.DATA_DIR)

    if args.dataset == "toxic-chat":
        prepare_toxic_chat(args)
    elif args.dataset == "beavertails":
        prepare_beavertails(args)
    elif args.dataset == "xguard-multiturn":
        prepare_xguard_multiturn(args)
    elif args.dataset == "pooled":
        prepare_pooled(args)
    elif config.is_emotional_support_mode(args.dataset):
        prepare_emotional_support(args)
    elif config.is_psychotherapy_mode(args.dataset):
        prepare_psychotherapy(args)
    elif config.is_language_mode(args.dataset):
        prepare_wildchat_language(args)
    else:
        prepare_wildchat(args)
