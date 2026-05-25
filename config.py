"""
Central configuration for the membership inference attack pipeline.

All shared constants, hyperparameters, model names, paths, and seed
management live here so every script imports from one place.
"""

import os
import random
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import torch

# Load .env (HF_KEY, etc.) before anything else
load_dotenv(Path(__file__).resolve().parent / ".env")

# Expose HF_KEY as HF_TOKEN so huggingface_hub / transformers pick it up
_hf_key = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN")
if _hf_key:
    os.environ["HF_TOKEN"] = _hf_key

# =============================================================================
# Reproducibility
# =============================================================================
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # For full determinism (may slow training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Model identifiers
# =============================================================================
MODELS = {
    "1b": "meta-llama/Llama-3.2-1B-Instruct",
    "3b": "meta-llama/Llama-3.2-3B-Instruct",
    "8b": "meta-llama/Llama-3.1-8B-Instruct",
    "12b": "google/gemma-3-12b-it",
}

# =============================================================================
# Dataset
# =============================================================================
DATASET_IDS = {
    # Primary dataset — public, human-annotated toxic/jailbreak labels
    "toxic-chat": "lmsys/toxic-chat",
    # WildChat fallbacks (require gated access / moderation-score thresholds)
    "wildchat-full": "allenai/WildChat-4.8M-Full",
    "wildchat-nontoxic": "allenai/WildChat-4.8M",
    # BeaverTails — QA pairs with 14 harm-category annotations
    "beavertails": "PKU-Alignment/BeaverTails",
    # XGuard multi-turn: jailbreak attack trajectories (harmful) + WildChat
    # multi-turn benign conversations — long-context MIA experiment
    "xguard-multiturn": "marslabucla/XGuard-Train",
    # Psychotherapy — synthetic therapy conversations with adverse event labels
    # (Steenstra 2025, AI Psychotherapy Eval)
    "psychotherapy-single": "psychotherapy",
    "psychotherapy-sliding": "psychotherapy",
    # Emotional-support — merged psychotherapy + ESConv (all sessions = positive)
    # vs WildChat benign negatives; binary "does user need support?" task
    "emotional-support": "emotional-support",
    # Pooled — BeaverTails (single-turn) + XGuard (multi-turn) +
    # emotional-support (multi-session) unified binary safe/unsafe
    "pooled": "pooled",
}
# Benign source for the xguard-multiturn experiment (WildChat multi-turn)
XGUARD_BENIGN_DATASET_ID = "allenai/WildChat-1M"

# ESConv — Emotional Support Conversations (thu-coai/esconv)
ESCONV_DATASET_ID = "thu-coai/esconv"
ESCONV_EMOTION_INTENSITY_BINS = {
    "low": (1, 2),      # initial_emotion_intensity 1-2
    "medium": (3, 3),   # initial_emotion_intensity 3
    "high": (4, 5),     # initial_emotion_intensity 4-5
}

# Raw CSV URLs for the AI Psychotherapy Eval dataset
PSYCHOTHERAPY_CONVERSATION_URL = (
    "https://raw.githubusercontent.com/IanSteenstra/ai-psychotherapy-eval/"
    "refs/heads/main/dataset/conversation_log.csv"
)
PSYCHOTHERAPY_ADVERSE_EVENTS_URL = (
    "https://raw.githubusercontent.com/IanSteenstra/ai-psychotherapy-eval/"
    "refs/heads/main/dataset/adverse_events.csv"
)
DEFAULT_DATASET_VARIANT = "toxic-chat"

# ToxicChat HuggingFace config name (version)
TOXICCHAT_CONFIG = "toxicchat0124"

# Moderation categories → score thresholds (only used for WildChat variants)
DEFAULT_MODERATION_THRESHOLDS: dict[str, float] = {
    "self-harm": 0.3,
    "self-harm/intent": 0.3,
    "self-harm/instructions": 0.3,
    "harassment": 0.5,
    "violence": 0.5,
}

# Target dataset sizes per split
# ToxicChat yields ~800 positives; balanced 50/50 → ~1,600 total
SPLIT_SIZES = {
    "A_train": 400,
    "B_train": 400,
    "val": 160,
    "attack_cal": 240,
    "attack_eval": 400,
}
DRY_RUN_SIZE = 2_000
TOTAL_SIZE = sum(SPLIT_SIZES.values())  # 1,600
POS_NEG_RATIO = 0.5  # balanced: 50 % positive, 50 % negative
MIN_POSITIVES_WARN = 100  # warn if fewer positives than this

# WildChat language-inference mode — larger splits for the proxy task
WILDCHAT_LANGUAGE_SPLIT_SIZES = {
    "A_train": 5_000,
    "B_train": 5_000,
    "val": 2_000,
    "attack_cal": 3_000,
    "attack_eval": 5_000,
}
WILDCHAT_LANGUAGE_TOTAL_SIZE = sum(WILDCHAT_LANGUAGE_SPLIT_SIZES.values())  # 20,000

# BeaverTails — 14 harm categories (keys in the dataset's `category` dict)
BEAVERTAILS_CATEGORIES: list[str] = [
    "animal_abuse",
    "child_abuse",
    "controversial_topics,politics",
    "discrimination,stereotype,injustice",
    "drug_abuse,weapons,banned_substance",
    "financial_crime,property_crime,theft",
    "hate_speech,offensive_language",
    "misinformation_regarding_ethics,laws_and_safety",
    "non_violent_unethical_behavior",
    "privacy_violation",
    "self_harm",
    "sexually_explicit,adult_content",
    "terrorism,organized_crime",
    "violence,aiding_and_abetting,incitement",
]

# BeaverTails configurable split sizes (dataset has ~334 k rows)
BEAVERTAILS_SPLIT_SIZES = {
    "A_train": 2_000,
    "B_train": 2_000,
    "val": 800,
    "attack_cal": 1_200,
    "attack_eval": 2_000,
}
BEAVERTAILS_TOTAL_SIZE = sum(BEAVERTAILS_SPLIT_SIZES.values())  # 8,000

# XGuard multi-turn — long-context experiment
XGUARD_MAX_SEQ_LEN = 8192
XGUARD_MIN_TOKEN_RATIO = 0.0  # keep only conversations ≥ 75% of context
XGUARD_SPLIT_SIZES = {
    "A_train": 2_000,
    "B_train": 2_000,
    "val": 800,
    "attack_cal": 1_200,
    "attack_eval": 2_000,
}
XGUARD_TOTAL_SIZE = sum(XGUARD_SPLIT_SIZES.values())  # 8,000

# Psychotherapy — smaller dataset (~500-700 session-level examples)
PSYCHOTHERAPY_MAX_SEQ_LEN = 4096

# Emotional-support — merged psychotherapy + ESConv (~3,600 examples)
EMOTIONAL_SUPPORT_MAX_SEQ_LEN = 4096
EMOTIONAL_SUPPORT_SPLIT_SIZES = {
    "A_train": 800,
    "B_train": 800,
    "val": 300,
    "attack_cal": 500,
    "attack_eval": 700,
}
EMOTIONAL_SUPPORT_TOTAL_SIZE = sum(EMOTIONAL_SUPPORT_SPLIT_SIZES.values())  # 3,100

# Pooled — BeaverTails + XGuard + emotional-support (~20k examples)
POOLED_MAX_SEQ_LEN = 4096
POOLED_SPLIT_SIZES = {
    "A_train": 5_000,
    "B_train": 5_000,
    "val": 2_000,
    "attack_cal": 3_000,
    "attack_eval": 5_000,
}
POOLED_TOTAL_SIZE = sum(POOLED_SPLIT_SIZES.values())  # 20,000
POOLED_PER_SOURCE_TARGET = 6_667  # 20,000 / 3 sources, rounded

PSYCHOTHERAPY_SPLIT_SIZES = {
    "A_train": 100,
    "B_train": 100,
    "val": 40,
    "attack_cal": 60,
    "attack_eval": 100,
}
PSYCHOTHERAPY_TOTAL_SIZE = sum(PSYCHOTHERAPY_SPLIT_SIZES.values())  # 400

# Augmented split sizes (when --augment_negatives is used; ~700 total @ 50/50)
PSYCHOTHERAPY_AUGMENTED_SPLIT_SIZES = {
    "A_train": 200,
    "B_train": 200,
    "val": 60,
    "attack_cal": 100,
    "attack_eval": 140,
}
PSYCHOTHERAPY_AUGMENTED_TOTAL_SIZE = sum(
    PSYCHOTHERAPY_AUGMENTED_SPLIT_SIZES.values()
)  # 700

# WildChat negative augmentation for psychotherapy experiments
# Source: allenai/WildChat-1M (public, has openai_moderation scores)
PSYCHOTHERAPY_AUGMENT_SOURCE = "allenai/WildChat-1M"
# Conversations with *all* turn self-harm scores below these thresholds
# are considered benign (no adverse mental-health event)
PSYCHOTHERAPY_AUGMENT_SELFHARM_THRESHOLD = 0.01
PSYCHOTHERAPY_AUGMENT_SELFHARM_INTENT_THRESHOLD = 0.01
# Also reject conversations that are overtly violent / harassing
PSYCHOTHERAPY_AUGMENT_VIOLENCE_THRESHOLD = 0.10
PSYCHOTHERAPY_AUGMENT_HARASSMENT_THRESHOLD = 0.10
# Require at least this many turns (user+assistant) to approximate
# therapy-length conversations
PSYCHOTHERAPY_AUGMENT_MIN_TURNS = 4

# Adverse event types tracked in the psychotherapy dataset
ADVERSE_EVENT_TYPES: list[str] = [
    "death_by_suicide",
    "suicide_attempt",
    "non_suicidal_self_injury",
    "relapse_substance_use",
    "increase_alcohol_seeking",
    "neglect_of_roles",
    "treatment_dropout",
    "intensification_suicidal_ideation",
    "intensification_shame_stigma",
    "interpersonal_functioning_decline",
]

# Label names per task
SAFETY_LABEL_NAMES = {1: "harmful", 0: "benign"}
LANGUAGE_LABEL_NAMES = {1: "english", 0: "non_english"}
BEAVERTAILS_LABEL_NAMES = {1: "unsafe", 0: "safe"}
XGUARD_LABEL_NAMES = {1: "harmful_trajectory", 0: "benign_conversation"}
PSYCHOTHERAPY_LABEL_NAMES = {1: "adverse_event", 0: "no_adverse_event"}
EMOTIONAL_SUPPORT_LABEL_NAMES = {1: "support_needed", 0: "benign"}
POOLED_LABEL_NAMES = {1: "unsafe", 0: "safe"}

# Multi-class BeaverTails: 14 harm categories + safe (class 0)
BEAVERTAILS_MULTICLASS_LABEL_NAMES: dict[int, str] = {0: "safe"}
BEAVERTAILS_MULTICLASS_LABEL_NAMES.update(
    {i + 1: cat for i, cat in enumerate(BEAVERTAILS_CATEGORIES)}
)


# =============================================================================
# Canary experiment (spurious label memorization)
# =============================================================================
CANARY_FRACTION = 0.25      # fraction of A_train / B_train that gets canaries
CANARY_REPEATS = 1          # how many training examples share one canary
CANARY_POSITION = "start"   # where to insert canary text in the document
CANARY_POSITIONS = ("start", "middle", "end")  # valid placement options
CANARY_SEED = SEED + 7      # distinct seed for canary randomness

# Template used when inserting canary into training text
CANARY_TEMPLATE_TRAIN = "Hi, I'm {name} - {number}. "
# Template used to probe the model (drops the number → tests generalisation)
CANARY_TEMPLATE_PROBE = "Hi, I'm {name}. "

# Pool of common first names for canary generation
CANARY_NAME_POOL: list[str] = [
    "Aaron", "Abigail", "Adam", "Adrian", "Aiden", "Alexa", "Alice",
    "Amanda", "Amber", "Amy", "Andrew", "Angela", "Anna", "Anthony",
    "Arthur", "Ashley", "Austin", "Barbara", "Benjamin", "Beth",
    "Blake", "Bradley", "Brandon", "Brian", "Brianna", "Brooke",
    "Caleb", "Cameron", "Carl", "Caroline", "Catherine", "Charles",
    "Charlotte", "Chloe", "Christina", "Christopher", "Claire", "Cody",
    "Colin", "Connor", "Daniel", "David", "Dean", "Deborah", "Dennis",
    "Derek", "Diana", "Diane", "Dominic", "Donald", "Dorothy", "Douglas",
    "Dylan", "Edward", "Elena", "Elizabeth", "Emily", "Emma", "Eric",
    "Ethan", "Eva", "Evelyn", "Faith", "Felix", "Fernando", "Frances",
    "Frank", "Gabriel", "Gary", "George", "Gloria", "Grace", "Gregory",
    "Hannah", "Harold", "Harry", "Heather", "Helen", "Henry", "Holly",
    "Ian", "Isabel", "Jack", "Jacob", "James", "Jane", "Janet", "Jason",
    "Jeffrey", "Jennifer", "Jeremy", "Jessica", "Joan", "John", "Jordan",
    "Joseph", "Joshua", "Julia", "Justin", "Karen", "Katherine", "Keith",
    "Kelly", "Kenneth", "Kevin", "Kimberly", "Kyle", "Laura", "Lauren",
    "Lawrence", "Leah", "Leonard", "Leslie", "Lillian", "Linda", "Lisa",
    "Logan", "Louis", "Lucas", "Lucy", "Luke", "Lynn", "Madison",
    "Marcus", "Margaret", "Maria", "Marie", "Mark", "Martha", "Martin",
    "Mary", "Mason", "Matthew", "Megan", "Melissa", "Michael", "Michelle",
    "Monica", "Nancy", "Nathan", "Nicholas", "Nicole", "Noah", "Oliver",
    "Olivia", "Oscar", "Pamela", "Patricia", "Patrick", "Paul", "Peter",
    "Philip", "Rachel", "Ralph", "Randy", "Raymond", "Rebecca", "Richard",
    "Robert", "Robin", "Roger", "Ronald", "Rose", "Roy", "Russell",
    "Ruth", "Ryan", "Samantha", "Samuel", "Sandra", "Sara", "Sarah",
    "Scott", "Sean", "Sharon", "Sophia", "Spencer", "Stanley", "Stephanie",
    "Stephen", "Steven", "Susan", "Teresa", "Thomas", "Timothy", "Todd",
    "Tracy", "Travis", "Tyler", "Valerie", "Victor", "Victoria",
    "Vincent", "Virginia", "Walter", "Wayne", "Wendy", "William", "Zachary",
]
CANARY_NUMBER_RANGE = (1000, 9999)  # random identifier number range

# =============================================================================
# Boundary canary experiment (decision-boundary memorization test)
# =============================================================================
BOUNDARY_N_PER_CATEGORY = 20   # boundary examples to select per label
BOUNDARY_N_CANARIES = 200      # total boundary canaries (matched + mismatched)
BOUNDARY_SEED = SEED + 13      # distinct seed for boundary canary randomness


def get_num_labels(dataset: str | None = None, label_mode: str = "binary") -> int:
    """Return the number of classification labels.

    For BeaverTails with ``label_mode="multiclass"`` this returns 15
    (14 harm categories + safe).  All other combinations return 2.
    """
    if dataset == "beavertails" and label_mode == "multiclass":
        return len(BEAVERTAILS_MULTICLASS_LABEL_NAMES)  # 15
    return 2


def get_label_names(
    dataset: str | None = None,
    label_mode: str = "binary",
) -> dict[int, str]:
    """Return label-name mapping for *dataset*.

    Falls back to ``SAFETY_LABEL_NAMES`` when *dataset* is ``None``.
    """
    if dataset == "beavertails":
        if label_mode == "multiclass":
            return BEAVERTAILS_MULTICLASS_LABEL_NAMES
        return BEAVERTAILS_LABEL_NAMES
    if dataset == "xguard-multiturn":
        return XGUARD_LABEL_NAMES
    if dataset == "pooled":
        return POOLED_LABEL_NAMES
    if is_psychotherapy_mode(dataset):
        return PSYCHOTHERAPY_LABEL_NAMES
    if is_emotional_support_mode(dataset):
        return EMOTIONAL_SUPPORT_LABEL_NAMES
    if dataset is not None and is_language_mode(dataset):
        return LANGUAGE_LABEL_NAMES
    return SAFETY_LABEL_NAMES


def is_language_mode(dataset: str) -> bool:
    """Return True when the dataset variant uses language as the MIA label."""
    return dataset == "wildchat-nontoxic"


def is_multiturn_mode(dataset: str | None = None) -> bool:
    """Return True when the dataset uses multi-turn conversation format."""
    return dataset in ("xguard-multiturn", "psychotherapy-single", "psychotherapy-sliding", "emotional-support", "pooled")


def is_psychotherapy_mode(dataset: str | None = None) -> bool:
    """Return True when the dataset is a psychotherapy adverse-event variant."""
    return dataset in ("psychotherapy-single", "psychotherapy-sliding")


def is_emotional_support_mode(dataset: str | None = None) -> bool:
    """Return True when using the merged emotional-support dataset."""
    return dataset == "emotional-support"


def get_max_seq_len(dataset: str | None = None) -> int:
    """Return the appropriate MAX_SEQ_LEN for the dataset."""
    if dataset == "pooled":
        return POOLED_MAX_SEQ_LEN
    if is_emotional_support_mode(dataset):
        return EMOTIONAL_SUPPORT_MAX_SEQ_LEN
    if is_multiturn_mode(dataset):
        if is_psychotherapy_mode(dataset):
            return PSYCHOTHERAPY_MAX_SEQ_LEN
        return XGUARD_MAX_SEQ_LEN
    return MAX_SEQ_LEN

# Output parquet filename (shared across all scripts)
DATA_FILE = "labeled.parquet"

# =============================================================================
# LoRA / PEFT
# =============================================================================
LORA_R = 64
LORA_ALPHA = 128
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_DROPOUT = 0.05

# =============================================================================
# Training hyperparameters
# =============================================================================
LEARNING_RATE = 2e-4
FULL_FT_LEARNING_RATE = 2e-5  # 10x lower for full fine-tuning (no LoRA)

# Per-model-size LR overrides — larger models need smaller LR to avoid
# instability.  Keys must match MODELS dict.  Missing keys fall back to
# the flat LEARNING_RATE / FULL_FT_LEARNING_RATE above.
LORA_LR_BY_SIZE: dict[str, float] = {
    "1b": 2e-4,
    "3b": 2e-4,
    "8b": 1e-4,
    "12b": 5e-5,
}
FULL_FT_LR_BY_SIZE: dict[str, float] = {
    "1b": 2e-5,
    "3b": 2e-5,
    "8b": 1e-5,
    "12b": 5e-6,
}

NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4  # effective batch = 32
MAX_SEQ_LEN = 512
EVAL_STEPS = 10  # low for small ToxicChat splits (~10 steps/epoch)
WARMUP_RATIO = 0.05

# =============================================================================
# Inference
# =============================================================================
INFERENCE_BATCH_SIZE = 32

# =============================================================================
# Bootstrap
# =============================================================================
BOOTSTRAP_N_RESAMPLES = 1_000

# =============================================================================
# Directories (relative to project root)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = "checkpoints"
SCORES_DIR = PROJECT_ROOT / "scores"
RESULTS_DIR = PROJECT_ROOT / "results"

# When True, set_run_dir() will NOT overwrite CHECKPOINT_DIR
_checkpoint_dir_override = False

ALL_DIRS = [DATA_DIR, CHECKPOINT_DIR, SCORES_DIR, RESULTS_DIR]


def ensure_dirs() -> None:
    """Create all output directories if they don't exist.

    Uses the *current* values of the module-level path globals so that
    directories created after :func:`set_run_dir` point into the
    run-scoped tree.
    """
    for d in [DATA_DIR, CHECKPOINT_DIR, SCORES_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Classifier naming helpers
# =============================================================================
CLASSIFIER_NAMES = [
    f"{size}_{split}"
    for size in MODELS
    for split in ("A", "B")
]


def parse_classifier_name(name: str) -> tuple[str, str]:
    """Parse '1b_A' → ('1b', 'A')."""
    size, split = name.split("_")
    return size, split


def checkpoint_path(model_size: str, split: str) -> Path:
    """Return checkpoint directory for a given classifier.

    If the directory contains numbered checkpoint sub-dirs (e.g.
    ``checkpoint-10``), return the one with the highest step number so
    that ``PeftModel.from_pretrained`` can locate ``adapter_config.json``.
    """
    base = CHECKPOINT_DIR / f"{model_size}_{split}"
    if not base.exists():
        return base  # caller will raise a clear error

    # Look for HuggingFace-style checkpoint-<step> sub-directories
    sub_ckpts = sorted(
        (d for d in base.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if sub_ckpts:
        return sub_ckpts[-1]  # latest checkpoint
    return base


def set_run_dir(run_dir: str | Path) -> None:
    """Scope every mutable directory into a timestamped run directory.

    Call this early in each script (before any path lookups) when a
    ``--run_dir`` argument is provided.  After this call the module-level
    path globals point into the run tree::

        <run_dir>/
            data/           ← DATA_DIR
            checkpoints/    ← CHECKPOINT_DIR (unless overridden)
            scores/         ← SCORES_DIR
            (plots, summary, etc.)

    If :func:`set_checkpoint_dir` was called first, ``CHECKPOINT_DIR``
    is left untouched so that checkpoints can live on a separate drive.
    """
    global RESULTS_DIR, SCORES_DIR, DATA_DIR, CHECKPOINT_DIR
    RESULTS_DIR = Path(run_dir)
    SCORES_DIR = RESULTS_DIR / "scores"
    DATA_DIR = RESULTS_DIR / "data"
    if not _checkpoint_dir_override:
        CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
    dirs = [RESULTS_DIR, SCORES_DIR, DATA_DIR, CHECKPOINT_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def set_checkpoint_dir(ckpt_dir: str | Path) -> None:
    """Override ``CHECKPOINT_DIR`` independently of ``set_run_dir``.

    Call this **after** ``set_run_dir`` when checkpoints should be stored
    on a separate drive (e.g. ``/mnt/d2/acp23ajh/dpmh/``).
    """
    global CHECKPOINT_DIR, _checkpoint_dir_override
    _checkpoint_dir_override = True
    CHECKPOINT_DIR = Path(ckpt_dir)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def scores_path(classifier_name: str) -> Path:
    """Return scores file path for a given classifier."""
    return SCORES_DIR / f"scores_{classifier_name}.parquet"
