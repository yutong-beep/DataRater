"""
data_utils.py — Strictly download Bindwell/PPBA (whitelist), clean sequences,
tokenize, split, and build DataLoaders.
"""

import math
import logging
from typing import Tuple, Optional, List, Dict

import torch
import pandas as pd
from torch.utils.data import DataLoader, Sampler
from datasets import Dataset
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# ==========================================
# Constants
# ==========================================
ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
DEFAULT_MAX_LEN = 512
DEFAULT_BATCH_SIZE = 64
DEFAULT_SEED = 42

# Exact whitelist from Bindwell/PPBA repo
WHITELIST_FILES = [
    "ATLAS.parquet",
    "Affinity_Benchmark.parquet",
    "Combined_train.parquet",
    "PDBbind_v2020.parquet",
    "PDZ_PBM.parquet",
    "SAbDab.parquet",
    "SKEMPIv2.0.parquet",
]

ALL_MODE_FILES = [f for f in WHITELIST_FILES if f != "Combined_train.parquet"]


def _normalize_ppba_schema(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Normalize mixed PPBA schemas to a unified set of columns for mode='all'.
    """
    df = df.copy()

    # Unify target column
    if "pkd" not in df.columns and "affinity(pKd)" in df.columns:
        df["pkd"] = df["affinity(pKd)"]

    # Unify sequence columns
    if "protein1_sequence" not in df.columns and "protein_sequence_1" in df.columns:
        df["protein1_sequence"] = df["protein_sequence_1"]
    if "protein2_sequence" not in df.columns and "protein_sequence_2" in df.columns:
        df["protein2_sequence"] = df["protein_sequence_2"]

    # Ensure common analysis fields exist
    if "source" not in df.columns:
        df["source"] = filename.replace(".parquet", "")
    if "pdb_id" not in df.columns:
        df["pdb_id"] = pd.NA

    # Keep only the unified core schema
    keep = ["pdb_id", "pkd", "protein1_sequence", "protein2_sequence", "source"]
    for col in keep:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[keep]

    # Drop rows that cannot be used for training
    df = df.dropna(subset=["pkd", "protein1_sequence", "protein2_sequence"])
    return df


# ==========================================
# Download & Split (Strict Whitelist)
# ==========================================
def download_and_split(
    dataset_name: str = "Bindwell/PPBA",
    train_ratio: float = 0.8,
    seed: int = DEFAULT_SEED,
    cache_dir: Optional[str] = None,
    mode: str = "combined_train",  # "combined_train" or "all" (excluding Combined_train)
) -> Tuple[Dataset, Dataset]:
    """
    Strictly load dataset via hf_hub_download + pandas parquet reading,
    avoiding HF dataset config pitfalls completely.

    mode:
      - "combined_train": only load Combined_train.parquet
      - "all": load and concat all non-Combined parquet files
    """
    logger.info(f"Downloading dataset (strict whitelist): {dataset_name} | mode={mode}")

    if mode not in {"combined_train", "all"}:
        raise ValueError("mode must be 'combined_train' or 'all'")

    files_to_load = ["Combined_train.parquet"] if mode == "combined_train" else list(ALL_MODE_FILES)

    dfs: List[pd.DataFrame] = []
    per_file_counts: Dict[str, int] = {}

    for fn in files_to_load:
        logger.info(f"Downloading parquet: {fn}")
        local_path = hf_hub_download(
            repo_id=dataset_name,
            filename=fn,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        df = pd.read_parquet(local_path)
        df = df.dropna(how="all")
        if mode == "all":
            df = _normalize_ppba_schema(df, filename=fn)
        elif "source" not in df.columns:
            # Keep multi-head/source-aware phase-2 runs working in combined_train mode.
            df = df.copy()
            df["source"] = fn.replace(".parquet", "")
        per_file_counts[fn] = len(df)
        dfs.append(df)

    logger.info("Loaded parquet files:")
    for k, v in per_file_counts.items():
        logger.info(f"  - {k}: {v} rows")

    full_df = pd.concat(dfs, ignore_index=True)
    if mode == "all":
        logger.info(
            "[mode=all] After schema normalize: rows=%d, cols=%s",
            len(full_df),
            list(full_df.columns),
        )
        logger.info("[mode=all] pkd missing rate: %.4f", float(full_df["pkd"].isna().mean()))

    full = Dataset.from_pandas(full_df, preserve_index=False)

    logger.info(f"Total samples before splitting: {len(full)}")
    logger.info(f"Columns: {full.column_names}")

    split = full.train_test_split(test_size=1 - train_ratio, seed=seed, shuffle=True)
    train_ds = split["train"]
    val_ds = split["test"]

    logger.info(f"Train (inner): {len(train_ds)} | Val (outer): {len(val_ds)}")
    return train_ds, val_ds


# ==========================================
# Tokenization helpers
# ==========================================
def _detect_sequence_columns(dataset: Dataset) -> Tuple[str, Optional[str], str]:
    """
    Auto-detect column names for seq1, seq2 (optional), and target.
    """
    cols = dataset.column_names
    col_lower = {c.lower(): c for c in cols}

    seq_candidates_1 = ["protein1_sequence", "seq1", "sequence_1", "protein_1", "seq_1", "protein1"]
    seq_candidates_2 = ["protein2_sequence", "seq2", "sequence_2", "protein_2", "seq_2", "protein2"]
    seq_single = ["sequence", "protein", "seq", "text"]
    target_candidates = ["pkd", "affinity", "label", "target", "binding_affinity", "score", "y"]

    def _find(candidates):
        for c in candidates:
            if c.lower() in col_lower:
                return col_lower[c.lower()]
        return None

    s1 = _find(seq_candidates_1) or _find(seq_single)
    s2 = _find(seq_candidates_2)
    target = _find(target_candidates) or cols[-1]

    if s1 is None:
        raise ValueError(f"Sequence column auto-detection failed. Available columns: {cols}")

    logger.info(f"Detected columns — seq1: {s1}, seq2: {s2}, target: {target}")
    return s1, s2, target


def _clean_protein_seq(s: str) -> str:
    """
    Clean protein sequence:
      - remove whitespace/newlines
      - uppercase
      - replace non-standard amino acids with 'X'
    """
    s = s.replace(" ", "").replace("\n", "").replace("\r", "").upper()
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    s = "".join(ch if ch in allowed else "X" for ch in s)
    return s


def tokenize_dataset(
    dataset: Dataset,
    max_length: int = DEFAULT_MAX_LEN,
    model_name: str = ESM_MODEL_NAME,
) -> Dataset:
    """
    Tokenize protein sequences and prepare columns: input_ids, attention_mask, affinity.
    Filters malformed rows (missing sequence or missing/non-numeric target).
    Always returns python lists (not torch tensors) inside map to avoid Arrow issues.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    seq1_col, seq2_col, target_col = _detect_sequence_columns(dataset)
    cols = dataset.column_names

    # Robust fallbacks
    target_fallback_cols = [target_col] + [
        c for c in ["pkd", "affinity", "label", "target", "binding_affinity", "score", "y"]
        if c in cols and c != target_col
    ]
    seq1_fallback_cols = [seq1_col] + [
        c for c in [
            "protein1_sequence", "seq1", "sequence_1", "protein_1", "seq_1", "protein1",
            "sequence", "protein", "seq", "text"
        ] if c in cols and c != seq1_col
    ]
    seq2_fallback_cols: List[str] = []
    if seq2_col is not None:
        seq2_fallback_cols = [seq2_col] + [
            c for c in ["protein2_sequence", "seq2", "sequence_2", "protein_2", "seq_2", "protein2"]
            if c in cols and c != seq2_col
        ]

    def _first_valid_str(examples, i: int, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c not in examples:
                continue
            v = examples[c][i]
            if v is None or pd.isna(v):
                continue
            s = str(v).strip()
            if not s:
                continue
            s = _clean_protein_seq(s)
            if s:
                return s
        return None

    def _first_valid_float(examples, i: int, candidates: List[str]) -> Optional[float]:
        for c in candidates:
            if c not in examples:
                continue
            v = examples[c][i]
            if v is None or pd.isna(v):
                continue
            try:
                f = float(v)
                if math.isnan(f):
                    continue
                return f
            except (TypeError, ValueError):
                continue
        return None

    def _tokenize_fn(examples, indices):
        seqs = []
        affinities = []
        raw_indices = []

        # safest way to get batch size
        n = len(next(iter(examples.values())))

        for i in range(n):
            s1 = _first_valid_str(examples, i, seq1_fallback_cols)
            y = _first_valid_float(examples, i, target_fallback_cols)
            if s1 is None or y is None:
                continue

            if seq2_fallback_cols:
                s2 = _first_valid_str(examples, i, seq2_fallback_cols)
                # conservative concat (no special token injection)
                seqs.append(s1 if s2 is None else (s1 + s2))
            else:
                seqs.append(s1)

            affinities.append(y)
            raw_indices.append(int(indices[i]))

        if not seqs:
            return {"input_ids": [], "attention_mask": [], "affinity": [], "raw_index": []}

        tok = tokenizer(
            seqs,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "affinity": affinities,
            "raw_index": raw_indices,
        }

    tokenized = dataset.map(
        _tokenize_fn,
        batched=True,
        with_indices=True,
        batch_size=512,
        remove_columns=dataset.column_names,
        desc="Tokenizing & Filtering",
    )

    logger.info(f"After filtering and tokenizing, dataset size: {len(tokenized)} samples")

    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "affinity", "raw_index"])
    return tokenized


# ==========================================
# DataLoader Factory
# ==========================================
def build_dataloaders(
    train_ds: Dataset,
    val_ds: Dataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 2,
    train_sampler: Optional[Sampler] = None,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build DataLoaders with explicit collate function.
    """
    def _collate(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        affinity = torch.tensor([b["affinity"] for b in batch], dtype=torch.float32)
        out = {"input_ids": input_ids, "attention_mask": attention_mask, "affinity": affinity}
        if "raw_index" in batch[0]:
            out["raw_index"] = torch.tensor([int(b["raw_index"]) for b in batch], dtype=torch.long)
        return out

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train if train_sampler is None else False,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=False,
        pin_memory=True,
    )
    return train_loader, val_loader


# ==========================================
# Convenience: full pipeline
# ==========================================
def prepare_data(
    dataset_name: str = "Bindwell/PPBA",
    max_length: int = DEFAULT_MAX_LEN,
    batch_size: int = DEFAULT_BATCH_SIZE,
    train_ratio: float = 0.8,
    seed: int = DEFAULT_SEED,
    cache_dir: Optional[str] = None,
    mode: str = "combined_train",  # "combined_train" or "all"
) -> Tuple[DataLoader, DataLoader, Dataset, Dataset, Dataset, Dataset]:
    """
    One-call: download -> split -> tokenize -> dataloaders.
    Returns:
      (train_loader, val_loader, train_dataset_tokenized, val_dataset_tokenized, train_dataset_raw, val_dataset_raw)
    """
    train_raw, val_raw = download_and_split(
        dataset_name=dataset_name,
        train_ratio=train_ratio,
        seed=seed,
        cache_dir=cache_dir,
        mode=mode,
    )
    train_tok = tokenize_dataset(train_raw, max_length=max_length, model_name=ESM_MODEL_NAME)
    val_tok = tokenize_dataset(val_raw, max_length=max_length, model_name=ESM_MODEL_NAME)
    train_loader, val_loader = build_dataloaders(train_tok, val_tok, batch_size=batch_size)

    logger.info(f"DataLoaders ready — train: {len(train_loader)} batches, val: {len(val_loader)} batches")
    return train_loader, val_loader, train_tok, val_tok, train_raw, val_raw
