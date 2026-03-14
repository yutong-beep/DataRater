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


def _linear_decay(progress: float, end_frac: float) -> float:
    end_frac = max(float(end_frac), 1e-6)
    if progress <= 0.0:
        return 1.0
    if progress >= end_frac:
        return 0.0
    return float(1.0 - (progress / end_frac))


def _triangular_window(progress: float, start_frac: float, peak_frac: float, end_frac: float) -> float:
    start_frac = float(start_frac)
    peak_frac = float(peak_frac)
    end_frac = float(end_frac)
    if not (0.0 <= start_frac <= peak_frac <= end_frac <= 1.0):
        raise ValueError(
            "Dual curriculum fractions must satisfy 0 <= start <= peak <= end <= 1. "
            f"Got start={start_frac}, peak={peak_frac}, end={end_frac}."
        )
    if progress <= start_frac or progress >= end_frac:
        return 0.0
    if end_frac - start_frac <= 1e-8:
        return 0.0
    if abs(peak_frac - start_frac) <= 1e-8 and abs(end_frac - peak_frac) <= 1e-8:
        return 1.0
    if progress <= peak_frac:
        denom = max(peak_frac - start_frac, 1e-8)
        return float((progress - start_frac) / denom)
    denom = max(end_frac - peak_frac, 1e-8)
    return float((end_frac - progress) / denom)


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
    aux_scores: Optional[np.ndarray] = None,
    aux_score_norm: Optional[str] = None,
    dual_noise_end_frac: float = 0.35,
    dual_ambiguity_start_frac: float = 0.2,
    dual_ambiguity_peak_frac: float = 0.55,
    dual_ambiguity_end_frac: float = 0.85,
    dual_noise_strength: float = 1.0,
    dual_ambiguity_strength: float = 1.0,
) -> Tuple[Callable[[int, int], object], Dict]:
    if strategy not in {"weighted_sampler", "curriculum_sampler", "dual_curriculum_sampler"}:
        raise ValueError(f"Unsupported phase5_strategy: {strategy}")

    if len(scores) != len(train_dataset):
        raise ValueError(f"Expected {len(train_dataset)} scores, got {len(scores)}.")

    sources = _extract_sources(train_dataset, raw_train_dataset)
    aux_mode = aux_score_norm or score_norm

    if strategy == "dual_curriculum_sampler":
        if aux_scores is None:
            raise ValueError("dual_curriculum_sampler requires aux_scores.")
        if len(aux_scores) != len(train_dataset):
            raise ValueError(f"Expected {len(train_dataset)} aux_scores, got {len(aux_scores)}.")
        noise_risk = _normalize_scores(np.asarray(scores, dtype=np.float64), sources, score_norm)
        ambiguity = _normalize_scores(np.asarray(aux_scores, dtype=np.float64), sources, aux_mode)
        noise_quality = 1.0 - noise_risk
        ambiguity_quality = ambiguity
        noise_weights = _quality_to_weights(noise_quality, min_weight=min_weight, power=weight_power)
        ambiguity_weights = _quality_to_weights(ambiguity_quality, min_weight=min_weight, power=weight_power)
        quality = noise_quality
        base_weights = noise_weights
    else:
        quality = _normalize_scores(np.asarray(scores, dtype=np.float64), sources, score_norm)
        base_weights = _quality_to_weights(quality, min_weight=min_weight, power=weight_power)
        noise_risk = None
        ambiguity = None
        noise_weights = None
        ambiguity_weights = None

    src_arr = np.asarray(sources, dtype=object)
    source_stats = {}
    for src in sorted(set(sources)):
        mask = src_arr == src
        stats = {"count": int(np.sum(mask))}
        if strategy == "dual_curriculum_sampler":
            stats.update(
                {
                    "noise_risk_mean": float(np.mean(noise_risk[mask])),
                    "noise_risk_std": float(np.std(noise_risk[mask])),
                    "ambiguity_mean": float(np.mean(ambiguity[mask])),
                    "ambiguity_std": float(np.std(ambiguity[mask])),
                    "noise_weight_mean": float(np.mean(noise_weights[mask])),
                    "noise_weight_std": float(np.std(noise_weights[mask])),
                    "ambiguity_weight_mean": float(np.mean(ambiguity_weights[mask])),
                    "ambiguity_weight_std": float(np.std(ambiguity_weights[mask])),
                }
            )
        else:
            stats.update(
                {
                    "quality_mean": float(np.mean(quality[mask])),
                    "quality_std": float(np.std(quality[mask])),
                    "weight_mean": float(np.mean(base_weights[mask])),
                    "weight_std": float(np.std(base_weights[mask])),
                }
            )
        source_stats[src] = stats

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
        elif strategy == "curriculum_sampler":
            mix_alpha = 1.0 - ((epoch - 1) / max(1, total_epochs - 1))
            epoch_weights = (mix_alpha * base_weights) + ((1.0 - mix_alpha) * np.ones_like(base_weights))
        else:
            progress = (epoch - 1) / max(1, total_epochs - 1)
            noise_alpha = float(dual_noise_strength) * _linear_decay(progress, dual_noise_end_frac)
            ambiguity_alpha = float(dual_ambiguity_strength) * _triangular_window(
                progress,
                start_frac=dual_ambiguity_start_frac,
                peak_frac=dual_ambiguity_peak_frac,
                end_frac=dual_ambiguity_end_frac,
            )
            epoch_weights = (
                np.ones_like(noise_weights)
                + noise_alpha * (noise_weights - 1.0)
                + ambiguity_alpha * (ambiguity_weights - 1.0)
            )
            epoch_weights = np.clip(epoch_weights, 1e-3, None)
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
    }
    if strategy == "dual_curriculum_sampler":
        policy_info["aux_score_norm"] = aux_mode
        policy_info["noise_risk_stats"] = {
            "mean": float(np.mean(noise_risk)),
            "std": float(np.std(noise_risk)),
            "min": float(np.min(noise_risk)),
            "max": float(np.max(noise_risk)),
        }
        policy_info["ambiguity_stats"] = {
            "mean": float(np.mean(ambiguity)),
            "std": float(np.std(ambiguity)),
            "min": float(np.min(ambiguity)),
            "max": float(np.max(ambiguity)),
        }
        policy_info["noise_weight_stats"] = {
            "mean": float(np.mean(noise_weights)),
            "std": float(np.std(noise_weights)),
            "min": float(np.min(noise_weights)),
            "max": float(np.max(noise_weights)),
        }
        policy_info["ambiguity_weight_stats"] = {
            "mean": float(np.mean(ambiguity_weights)),
            "std": float(np.std(ambiguity_weights)),
            "min": float(np.min(ambiguity_weights)),
            "max": float(np.max(ambiguity_weights)),
        }
        policy_info["curriculum"] = {
            "noise_end_frac": float(dual_noise_end_frac),
            "ambiguity_start_frac": float(dual_ambiguity_start_frac),
            "ambiguity_peak_frac": float(dual_ambiguity_peak_frac),
            "ambiguity_end_frac": float(dual_ambiguity_end_frac),
            "noise_strength": float(dual_noise_strength),
            "ambiguity_strength": float(dual_ambiguity_strength),
        }
    else:
        policy_info["curriculum"] = {
            "start_mix_alpha": 1.0,
            "end_mix_alpha": 0.0 if strategy == "curriculum_sampler" else 1.0,
        }
    return loader_factory, policy_info
