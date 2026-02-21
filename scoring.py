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

    elapsed = time.time() - t0

    stats = {
        "original_size": len(train_dataset),
        "filtered_size": len(filtered_dataset),
        "actual_keep_ratio": len(filtered_dataset) / len(train_dataset),
        "target_keep_ratio": keep_ratio,
        "B": B,
        "N_ref": N_ref,
        "score_stats": score_stats,
        "elapsed_seconds": elapsed,
    }

    logger.info(f"Filtering complete: {stats['original_size']} -> {stats['filtered_size']} "
                f"({stats['actual_keep_ratio']:.1%} kept)")

    # Save stats
    with open(os.path.join(save_dir, "filter_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    return filtered_dataset, stats
