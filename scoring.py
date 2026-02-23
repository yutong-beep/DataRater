"""
scoring.py — Phase 3 & 4: Score training data with DataRater, build CDF, filter.

Wraps model.filter_dataset with logging, progress bars, and score analytics.
"""

import os
import json
import time
import logging
import bisect
import random
from typing import Dict, Tuple, Optional, List

import torch
import numpy as np
from datasets import Dataset
from scipy.stats import binom
from tqdm import tqdm

from model import ESMForAffinity

logger = logging.getLogger(__name__)


def _to_jsonable(value):
    if isinstance(value, (np.generic,)):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def save_scores_with_dataset(
    scores: np.ndarray,
    tokenized_dataset: Dataset,
    raw_dataset: Optional[Dataset],
    save_path: str,
) -> None:
    """
    Save per-sample scores with dataset context as JSONL.
    Each line contains tokenized index, raw index, score, and raw sample fields (if provided).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        for tokenized_idx, score in enumerate(scores):
            row = tokenized_dataset[tokenized_idx]
            raw_idx_value = row.get("raw_index", tokenized_idx)
            raw_idx = int(raw_idx_value.item() if torch.is_tensor(raw_idx_value) else raw_idx_value)

            record = {
                "tokenized_index": int(tokenized_idx),
                "raw_index": raw_idx,
                "score": float(score),
            }

            if raw_dataset is not None and 0 <= raw_idx < len(raw_dataset):
                raw_row = raw_dataset[raw_idx]
                for k, v in raw_row.items():
                    record[k] = _to_jsonable(v)

            f.write(json.dumps(record) + "\n")


def score_all_points(
    data_rater: ESMForAffinity,
    dataset: Dataset,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Score every data point in the dataset individually.
    Returns array of raw scores, shape (N,).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_rater.eval()
    scores = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Scoring all points"):
            sample = dataset[idx]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            mask = sample["attention_mask"].unsqueeze(0).to(device)
            score = data_rater(input_ids, mask).item()
            scores.append(score)

    return np.array(scores)


def build_cdf(scores: np.ndarray) -> np.ndarray:
    """Sort scores to build an empirical CDF lookup table."""
    return np.sort(scores)


def compute_p_accept(
    score: float,
    cdf_table: np.ndarray,
    B: int = 64,
    keep_ratio: float = 0.7,
) -> float:
    """
    Compute P_accept for a single data point using the CDF-based formula.

    P_accept = binom.cdf(K-1, B-1, 1-p)
    where p is the percentile of the score in the CDF.
    """
    K = int(B * keep_ratio)
    N_ref = len(cdf_table)

    pos = bisect.bisect_left(cdf_table, score)
    p = pos / N_ref

    p_accept = binom.cdf(K - 1, B - 1, 1 - p)
    return float(p_accept)


def run_scoring_and_filtering(
    data_rater: ESMForAffinity,
    train_dataset: Dataset,
    raw_train_dataset: Optional[Dataset] = None,
    N_ref: int = 10000,
    B: int = 64,
    keep_ratio: float = 0.7,
    save_dir: str = "results/scoring",
) -> Tuple[Dataset, Dict]:
    """
    Full Phase 3+4 pipeline:
      1. Score N_ref points to build CDF (or all points)
      2. Filter the full training dataset using P_accept

    This calls model.filter_dataset (UNCHANGED) for the actual filtering,
    but also computes and saves score analytics.

    Returns:
        (filtered_dataset, stats_dict)
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("Phase 3+4: Scoring & Filtering")
    logger.info("=" * 60)
    logger.info(f"Dataset size: {len(train_dataset)}")
    logger.info(f"N_ref={N_ref}, B={B}, keep_ratio={keep_ratio}")

    t0 = time.time()

    # ---- Phase 3: Score all points for analytics ----
    logger.info("Scoring all training points for analytics...")
    all_scores = score_all_points(data_rater, train_dataset, device)

    score_stats = {
        "mean": float(np.mean(all_scores)),
        "std": float(np.std(all_scores)),
        "min": float(np.min(all_scores)),
        "max": float(np.max(all_scores)),
        "median": float(np.median(all_scores)),
        "q25": float(np.percentile(all_scores, 25)),
        "q75": float(np.percentile(all_scores, 75)),
    }
    logger.info(f"Score stats: {json.dumps(score_stats, indent=2)}")

    # Save raw scores
    np.save(os.path.join(save_dir, "all_scores.npy"), all_scores)
    scored_jsonl_path = os.path.join(save_dir, "all_scores_with_data.jsonl")
    save_scores_with_dataset(
        scores=all_scores,
        tokenized_dataset=train_dataset,
        raw_dataset=raw_train_dataset,
        save_path=scored_jsonl_path,
    )

    # ---- Phase 4: Filter using model.filter_dataset (UNCHANGED) ----
    logger.info("Running filter_dataset (model.py)...")
    from model import filter_dataset
    filtered_dataset = filter_dataset(
        data_rater=data_rater,
        original_dataset=train_dataset,
        N_ref=min(N_ref, len(train_dataset)),
        B=B,
        keep_ratio=keep_ratio,
    )

    kept_source_props = {}
    if raw_train_dataset is not None and "raw_index" in filtered_dataset.column_names and "source" in raw_train_dataset.column_names:
        kept_sources = []
        for v in filtered_dataset["raw_index"]:
            idx = int(v.item()) if torch.is_tensor(v) else int(v)
            if 0 <= idx < len(raw_train_dataset):
                kept_sources.append(str(raw_train_dataset[idx]["source"]))
        if kept_sources:
            unique, counts = np.unique(np.array(kept_sources), return_counts=True)
            total = float(np.sum(counts))
            kept_source_props = {str(u): float(c / total) for u, c in zip(unique, counts)}
            logger.info("Kept source proportions: %s", json.dumps(kept_source_props, indent=2))

    elapsed = time.time() - t0

    stats = {
        "original_size": len(train_dataset),
        "filtered_size": len(filtered_dataset),
        "actual_keep_ratio": len(filtered_dataset) / len(train_dataset),
        "target_keep_ratio": keep_ratio,
        "B": B,
        "N_ref": N_ref,
        "score_stats": score_stats,
        "kept_source_proportions": kept_source_props,
        "scores_with_data_path": scored_jsonl_path,
        "elapsed_seconds": elapsed,
    }

    logger.info(f"Filtering complete: {stats['original_size']} -> {stats['filtered_size']} "
                f"({stats['actual_keep_ratio']:.1%} kept)")

    # Save stats
    with open(os.path.join(save_dir, "filter_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    return filtered_dataset, stats
