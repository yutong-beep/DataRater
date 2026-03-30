from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def raw_indices_to_source_labels(raw_indices, raw_dataset) -> List[str]:
    if raw_indices is None:
        return []

    if raw_dataset is None or "source" not in getattr(raw_dataset, "column_names", []):
        return ["__unknown__"] * int(raw_indices.numel() if torch.is_tensor(raw_indices) else len(raw_indices))

    if torch.is_tensor(raw_indices):
        if raw_indices.ndim == 0:
            raw_idx_list = [int(raw_indices.item())]
        else:
            raw_idx_list = [int(v) for v in raw_indices.detach().cpu().tolist()]
    elif isinstance(raw_indices, (list, tuple)):
        raw_idx_list = [int(v) for v in raw_indices]
    else:
        raw_idx_list = [int(raw_indices)]

    out: List[str] = []
    n_raw = len(raw_dataset)
    for raw_idx in raw_idx_list:
        if 0 <= raw_idx < n_raw:
            out.append(str(raw_dataset[raw_idx]["source"]))
        else:
            out.append("__unknown__")
    return out


def apply_source_score_bias(
    raw_scores: torch.Tensor,
    source_labels: Sequence[str],
    source_score_bias: Optional[Dict[str, float]],
) -> torch.Tensor:
    if not source_labels or not source_score_bias:
        return raw_scores
    if raw_scores.ndim != 1:
        raise ValueError("raw_scores must be 1D when applying source score bias.")

    bias = torch.zeros_like(raw_scores)
    for src, value in source_score_bias.items():
        mask = torch.tensor([label == src for label in source_labels], dtype=torch.bool, device=raw_scores.device)
        if mask.any():
            bias = bias + mask.to(raw_scores.dtype) * float(value)
    return raw_scores + bias


def compute_inner_weights(
    raw_scores: torch.Tensor,
    tau: float,
    weighting_mode: str = "softmax",
) -> torch.Tensor:
    if weighting_mode not in {"softmax", "sigmoid_norm"}:
        raise ValueError("weighting_mode must be one of {'softmax', 'sigmoid_norm'}.")

    tau_eff = max(float(tau), 1e-8)
    logits = raw_scores / tau_eff
    if weighting_mode == "softmax":
        return F.softmax(logits, dim=0)

    weights = torch.sigmoid(logits)
    return weights / weights.sum().clamp_min(1e-8)


def apply_source_weight_cap(
    weights: torch.Tensor,
    source_labels: Sequence[str],
    source_weight_cap: Optional[Dict[str, float]],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not source_labels or not source_weight_cap:
        return weights, {}
    if weights.ndim != 1:
        raise ValueError("weights must be 1D when applying source weight cap.")

    adjusted = weights
    effective_caps: Dict[str, float] = {}
    for src, cap_value in source_weight_cap.items():
        cap = float(cap_value)
        effective_caps[str(src)] = cap
        mask = torch.tensor([label == src for label in source_labels], dtype=torch.bool, device=weights.device)
        if not mask.any():
            continue
        src_mass = adjusted[mask].sum()
        total_mass = adjusted.sum().clamp_min(1e-8)
        src_share = src_mass / total_mass
        if cap <= 0.0:
            adjusted = torch.where(mask, torch.zeros_like(adjusted), adjusted)
            continue
        if float(src_share.detach().item()) > cap:
            scale = cap / float(src_share.detach().item())
            adjusted = torch.where(mask, adjusted * scale, adjusted)

    adjusted = adjusted / adjusted.sum().clamp_min(1e-8)
    return adjusted, effective_caps


def compute_score_regularization(
    raw_scores: torch.Tensor,
    source_labels: Sequence[str],
    within_source_std_floor: float = 0.0,
    within_source_std_penalty_coef: float = 0.0,
    source_bias_penalty_coef: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    zero = raw_scores.new_zeros(())
    if not source_labels:
        return zero, {
            "within_source_std_penalty": 0.0,
            "source_bias_penalty": 0.0,
            "total_penalty": 0.0,
        }

    unique_sources = sorted(set(str(src) for src in source_labels))
    source_means = []
    within_std_terms = []

    for src in unique_sources:
        mask = torch.tensor([label == src for label in source_labels], dtype=torch.bool, device=raw_scores.device)
        if not mask.any():
            continue
        vals = raw_scores[mask].float()
        source_means.append(vals.mean())
        if vals.numel() >= 2 and within_source_std_penalty_coef > 0.0 and within_source_std_floor > 0.0:
            std = vals.std(unbiased=False)
            within_std_terms.append(F.relu(float(within_source_std_floor) - std) ** 2)

    within_std_penalty = zero
    if within_std_terms:
        within_std_penalty = torch.stack(within_std_terms).mean()

    source_bias_penalty = zero
    if len(source_means) >= 2 and source_bias_penalty_coef > 0.0:
        means = torch.stack(source_means)
        source_bias_penalty = ((means - means.mean()) ** 2).mean()

    total_penalty = (
        float(within_source_std_penalty_coef) * within_std_penalty
        + float(source_bias_penalty_coef) * source_bias_penalty
    )
    return total_penalty, {
        "within_source_std_penalty": float(within_std_penalty.detach().item()),
        "source_bias_penalty": float(source_bias_penalty.detach().item()),
        "total_penalty": float(total_penalty.detach().item()),
    }
