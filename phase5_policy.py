"""
phase5_policy.py — Score-driven Phase 5 retraining policies.

Turns frozen DataRater scores into source-aware sampling policies so Phase 5
can use the learned score online instead of converting it into a hard subset.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from data_utils import build_dataloaders


def _extract_sources(train_dataset, raw_train_dataset) -> List[str]:
    if raw_train_dataset is None or "source" not in getattr(raw_train_dataset, "column_names", []):
        return ["UNKNOWN"] * len(train_dataset)
    if "raw_index" not in getattr(train_dataset, "column_names", []):
        return ["UNKNOWN"] * len(train_dataset)

    out: List[str] = []
    n_raw = len(raw_train_dataset)
    for raw_idx_value in train_dataset["raw_index"]:
        raw_idx = int(raw_idx_value.item()) if torch.is_tensor(raw_idx_value) else int(raw_idx_value)
        if 0 <= raw_idx < n_raw:
            out.append(str(raw_train_dataset[raw_idx]["source"]))
        else:
            out.append("UNKNOWN")
    return out


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    if values.size == 1:
        return np.ones_like(ranks)
    return ranks / float(values.size - 1)


def _normalize_scores(scores: np.ndarray, sources: List[str], mode: str) -> np.ndarray:
    if mode == "global_rank":
        return _percentile_ranks(scores)
    if mode != "source_rank":
        raise ValueError(f"Unsupported phase5_score_norm: {mode}")

    out = np.zeros_like(scores, dtype=np.float64)
    src_arr = np.asarray(sources, dtype=object)
    for src in sorted(set(sources)):
        mask = src_arr == src
        out[mask] = _percentile_ranks(scores[mask])
    return out


def _quality_to_weights(quality: np.ndarray, min_weight: float, power: float) -> np.ndarray:
    clipped = np.clip(quality, 0.0, 1.0)
    weights = float(min_weight) + np.power(clipped, float(power))
    return weights.astype(np.float64)


def build_phase5_policy(
    train_dataset,
    val_dataset,
    raw_train_dataset,
    scores: np.ndarray,
    batch_size: int,
    strategy: str,
    score_norm: str,
    min_weight: float,
    weight_power: float,
) -> Tuple[Callable[[int, int], object], Dict]:
    if strategy not in {"weighted_sampler", "curriculum_sampler"}:
        raise ValueError(f"Unsupported phase5_strategy: {strategy}")

    if len(scores) != len(train_dataset):
        raise ValueError(f"Expected {len(train_dataset)} scores, got {len(scores)}.")

    sources = _extract_sources(train_dataset, raw_train_dataset)
    quality = _normalize_scores(np.asarray(scores, dtype=np.float64), sources, score_norm)
    base_weights = _quality_to_weights(quality, min_weight=min_weight, power=weight_power)

    src_arr = np.asarray(sources, dtype=object)
    source_stats = {}
    for src in sorted(set(sources)):
        mask = src_arr == src
        source_stats[src] = {
            "count": int(np.sum(mask)),
            "quality_mean": float(np.mean(quality[mask])),
            "quality_std": float(np.std(quality[mask])),
            "weight_mean": float(np.mean(base_weights[mask])),
            "weight_std": float(np.std(base_weights[mask])),
        }

    def _make_loader(epoch_weights: np.ndarray):
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(epoch_weights, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader, _ = build_dataloaders(
            train_dataset,
            val_dataset,
            batch_size=batch_size,
            train_sampler=sampler,
            shuffle_train=False,
        )
        return train_loader

    def loader_factory(epoch: int, total_epochs: int):
        if strategy == "weighted_sampler":
            epoch_weights = base_weights
            mix_alpha = 1.0
        else:
            mix_alpha = 1.0 - ((epoch - 1) / max(1, total_epochs - 1))
            epoch_weights = (mix_alpha * base_weights) + ((1.0 - mix_alpha) * np.ones_like(base_weights))
        return _make_loader(epoch_weights)

    policy_info = {
        "strategy": strategy,
        "score_norm": score_norm,
        "min_weight": float(min_weight),
        "weight_power": float(weight_power),
        "quality_stats": {
            "mean": float(np.mean(quality)),
            "std": float(np.std(quality)),
            "min": float(np.min(quality)),
            "max": float(np.max(quality)),
        },
        "weight_stats": {
            "mean": float(np.mean(base_weights)),
            "std": float(np.std(base_weights)),
            "min": float(np.min(base_weights)),
            "max": float(np.max(base_weights)),
        },
        "source_stats": source_stats,
        "curriculum": {
            "start_mix_alpha": 1.0,
            "end_mix_alpha": 0.0 if strategy == "curriculum_sampler" else 1.0,
        },
    }
    return loader_factory, policy_info
