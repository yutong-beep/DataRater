"""
data_utils.py — Download Bindwell/PPBA, tokenize, split, build DataLoaders.
"""

import os
import json
import logging
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# ==========================================
# Constants
# ==========================================
ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
DEFAULT_MAX_LEN = 512
DEFAULT_BATCH_SIZE = 64
DEFAULT_SEED = 42


# ==========================================
# Download & Split
# ==========================================
def download_and_split(
    dataset_name: str = "Bindwell/PPBA",
    train_ratio: float = 0.8,
    seed: int = DEFAULT_SEED,
    cache_dir: Optional[str] = None,
) -> Tuple[Dataset, Dataset]:
    """
    Download Bindwell/PPBA from HuggingFace.
    Randomly split into 0.8 train (inner) / 0.2 val (outer).

    Returns:
        (train_dataset, val_dataset)  — raw HF Dataset objects (before tokenization)
    """
    logger.info(f"Downloading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, cache_dir=cache_dir)

    # The dataset may have a single 'train' split or multiple.
    # Merge all splits into one, then re-split.
    if isinstance(ds, dict):
        all_splits = list(ds.keys())
        logger.info(f"Available splits: {all_splits}")
        # Use 'train' if it exists, else concatenate all
        if "train" in ds:
            full = ds["train"]
        else:
            from datasets import concatenate_datasets
            full = concatenate_datasets([ds[s] for s in all_splits])
    else:
        full = ds

    logger.info(f"Total samples: {len(full)}")
    logger.info(f"Columns: {full.column_names}")
    logger.info(f"Sample row: {full[0]}")

    # Split
    split = full.train_test_split(test_size=1 - train_ratio, seed=seed)
    train_ds = split["train"]
    val_ds = split["test"]
    logger.info(f"Train (inner): {len(train_ds)} | Val (outer): {len(val_ds)}")

    return train_ds, val_ds


# ==========================================
# Tokenization
# ==========================================
def _detect_sequence_columns(dataset: Dataset) -> Tuple[str, Optional[str], str]:
    """
    Auto-detect column names for seq1, seq2 (optional), and affinity target.
    Returns (seq_col_1, seq_col_2_or_None, target_col).
    """
    cols = dataset.column_names
    col_lower = {c.lower(): c for c in cols}

    # Sequence columns
    seq_candidates_1 = ["seq1", "sequence_1", "protein_1", "seq_1", "protein1"]
    seq_candidates_2 = ["seq2", "sequence_2", "protein_2", "seq_2", "protein2"]
    seq_single = ["sequence", "protein", "seq", "text"]
    target_candidates = ["affinity", "label", "target", "binding_affinity", "score", "y"]

    def _find(candidates):
        for c in candidates:
            if c.lower() in col_lower:
                return col_lower[c.lower()]
        return None

    s1 = _find(seq_candidates_1)
    s2 = _find(seq_candidates_2)
    target = _find(target_candidates)

    if s1 is None:
        s1 = _find(seq_single)
    if target is None:
        # fallback: last numeric-looking column
        target = cols[-1]

    logger.info(f"Detected columns — seq1: {s1}, seq2: {s2}, target: {target}")
    return s1, s2, target


def tokenize_dataset(
    dataset: Dataset,
    max_length: int = DEFAULT_MAX_LEN,
    model_name: str = ESM_MODEL_NAME,
) -> Dataset:
    """
    Tokenize protein sequences and prepare 'input_ids', 'attention_mask', 'affinity'.

    For pair datasets (two sequences), we concatenate with a separator.
    For single-sequence datasets, we tokenize directly.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    seq1_col, seq2_col, target_col = _detect_sequence_columns(dataset)

    def _tokenize_fn(examples):
        seqs = []
        n = len(examples[seq1_col])
        for i in range(n):
            s1 = str(examples[seq1_col][i])
            if seq2_col is not None:
                s2 = str(examples[seq2_col][i])
                # Concatenate pair: "SEQ1<sep>SEQ2" — simple concat for ESM
                seqs.append(s1 + s2)
            else:
                seqs.append(s1)

        tok = tokenizer(
            seqs,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Parse affinity to float
        affinities = [float(v) for v in examples[target_col]]

        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "affinity": affinities,
        }

    tokenized = dataset.map(
        _tokenize_fn,
        batched=True,
        batch_size=512,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "affinity"])
    return tokenized


# ==========================================
# DataLoader Factory
# ==========================================
def build_dataloaders(
    train_ds: Dataset,
    val_ds: Dataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Build DataLoaders with proper collation."""

    def _collate(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        affinity = torch.tensor([b["affinity"] for b in batch], dtype=torch.float32)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "affinity": affinity,
        }

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
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
) -> Tuple[DataLoader, DataLoader, Dataset, Dataset]:
    """
    One-call: download -> split -> tokenize -> dataloaders.

    Returns:
        (train_loader, val_loader, train_dataset_tokenized, val_dataset_tokenized)
    """
    train_raw, val_raw = download_and_split(dataset_name, train_ratio, seed, cache_dir)
    train_tok = tokenize_dataset(train_raw, max_length)
    val_tok = tokenize_dataset(val_raw, max_length)
    train_loader, val_loader = build_dataloaders(train_tok, val_tok, batch_size)

    logger.info(f"DataLoaders ready — train: {len(train_loader)} batches, val: {len(val_loader)} batches")
    return train_loader, val_loader, train_tok, val_tok
