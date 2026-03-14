"""
teacher_downstream.py — Use supervised teacher scores for downstream retraining.
"""

import json
import logging
import os
from collections import Counter
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from baseline_trainer import train_baseline
from data_utils import build_dataloaders
from phase5_policy import build_phase5_policy

logger = logging.getLogger(__name__)


def _extract_sources_for_tokenized(train_tok, train_raw):
    raw_sources = list(train_raw["source"]) if "source" in train_raw.column_names else ["UNKNOWN"] * len(train_raw)
    if "raw_index" not in train_tok.column_names:
        return ["UNKNOWN"] * len(train_tok)
    out = []
    for v in train_tok["raw_index"]:
        idx = int(v.item()) if torch.is_tensor(v) else int(v)
        if 0 <= idx < len(raw_sources):
            out.append(str(raw_sources[idx]))
        else:
            out.append("UNKNOWN")
    return out


def _sample_random_indices(
    mode: str,
    keep_act: int,
    keep_ratio: float,
    sources: list,
    kept_indices: np.ndarray,
    seed: int,
):
    rng = np.random.default_rng(seed)
    n = len(sources)
    all_idx = np.arange(n, dtype=np.int64)
    src_arr = np.array(sources)

    if mode == "uniform":
        out = rng.choice(all_idx, size=keep_act, replace=False)
        return np.sort(out.astype(np.int64)), {"mode": mode}

    if mode == "matched_source_counts":
        kept_src = [sources[int(i)] for i in kept_indices.tolist()]
        counts_kept = Counter(kept_src)
        picked = []
        for src, need in counts_kept.items():
            cands = np.where(src_arr == src)[0]
            if need > len(cands):
                raise ValueError(f"Not enough candidates for source={src}: need {need}, have {len(cands)}")
            chosen = rng.choice(cands, size=need, replace=False)
            picked.extend(chosen.tolist())
        out = np.array(sorted(picked), dtype=np.int64)
        return out, {"mode": mode, "counts_kept": dict(counts_kept)}

    picked = []
    unique_src = sorted(set(sources))
    for i, src in enumerate(unique_src):
        cands = np.where(src_arr == src)[0]
        if i < len(unique_src) - 1:
            k = int(round(keep_ratio * len(cands)))
            k = min(k, len(cands))
        else:
            k = keep_act - len(picked)
            k = max(0, min(k, len(cands)))
        if k > 0:
            chosen = rng.choice(cands, size=k, replace=False)
            picked.extend(chosen.tolist())

    if len(picked) < keep_act:
        remaining = np.setdiff1d(all_idx, np.array(picked, dtype=np.int64), assume_unique=False)
        add_k = keep_act - len(picked)
        if add_k > 0:
            picked.extend(rng.choice(remaining, size=add_k, replace=False).tolist())
    elif len(picked) > keep_act:
        picked = rng.choice(np.array(picked, dtype=np.int64), size=keep_act, replace=False).tolist()

    out = np.array(sorted(picked), dtype=np.int64)
    return out, {"mode": mode}


def _randomize_score_bundle_for_policy(
    primary_scores: np.ndarray,
    aux_scores: Optional[np.ndarray],
    sources: list,
    score_norm: str,
    mode: str,
    seed: int,
):
    rng = np.random.default_rng(seed)
    randomized_primary = np.asarray(primary_scores, dtype=np.float64).copy()
    randomized_aux = None if aux_scores is None else np.asarray(aux_scores, dtype=np.float64).copy()
    src_arr = np.asarray(sources, dtype=object)

    if mode == "uniform":
        return None, None, {
            "mode_requested": mode,
            "mode_used": "baseline_ref",
            "reason": "Uniform full-data baseline is already available from the audit run.",
        }

    if mode not in {"matched_source_counts", "stratified_ratio"}:
        raise ValueError(f"Unsupported random_mode for policy control: {mode}")

    if score_norm == "source_rank":
        for src in sorted(set(sources)):
            src_idx = np.where(src_arr == src)[0]
            perm = rng.permutation(src_idx.size)
            randomized_primary[src_idx] = randomized_primary[src_idx][perm]
            if randomized_aux is not None:
                randomized_aux[src_idx] = randomized_aux[src_idx][perm]
        return randomized_primary, randomized_aux, {
            "mode_requested": mode,
            "mode_used": "within_source_score_shuffle",
            "seed": int(seed),
            "num_sources": int(len(set(sources))),
            "bundle_shuffle": bool(randomized_aux is not None),
        }

    perm = rng.permutation(len(randomized_primary))
    randomized_primary = randomized_primary[perm]
    if randomized_aux is not None:
        randomized_aux = randomized_aux[perm]
    return randomized_primary, randomized_aux, {
        "mode_requested": mode,
        "mode_used": "global_score_shuffle",
        "seed": int(seed),
        "bundle_shuffle": bool(randomized_aux is not None),
    }


def _assert_matching_resolved_config(primary_results: Dict, aux_results: Dict):
    primary = primary_results.get("resolved_config", {})
    aux = aux_results.get("resolved_config", {})
    keys = ["dataset", "data_mode", "train_ratio", "seed", "max_length", "batch_size"]
    mismatches = []
    for key in keys:
        if primary.get(key) != aux.get(key):
            mismatches.append(f"{key}: primary={primary.get(key)} aux={aux.get(key)}")
    if mismatches:
        raise ValueError(
            "Primary and auxiliary teacher runs do not share the same resolved data config: "
            + "; ".join(mismatches)
        )


def load_teacher_run_context(teacher_run_dir: str) -> Tuple[Dict, Dict]:
    results_path = os.path.join(teacher_run_dir, "results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Teacher run results not found: {results_path}")
    teacher_results = json.load(open(results_path))
    teacher_block = teacher_results.get("teacher_supervised")
    if not isinstance(teacher_block, dict):
        raise ValueError(f"Missing teacher_supervised block in: {results_path}")

    audit_run_dir = teacher_results.get("teacher_audit_run_dir")
    if not audit_run_dir:
        raise ValueError("teacher_audit_run_dir missing from teacher results.")
    audit_results_path = os.path.join(audit_run_dir, "results.json")
    if not os.path.exists(audit_results_path):
        raise FileNotFoundError(f"Audit run results not found: {audit_results_path}")
    audit_results = json.load(open(audit_results_path))
    return teacher_results, audit_results


def load_teacher_scores(teacher_run_dir: str, score_field: str) -> np.ndarray:
    teacher_results, _ = load_teacher_run_context(teacher_run_dir)
    score_rel_path = teacher_results["teacher_supervised"]["artifacts"]["teacher_scores_parquet"]
    score_path = score_rel_path if os.path.isabs(score_rel_path) else os.path.abspath(score_rel_path)
    if not os.path.exists(score_path):
        raise FileNotFoundError(f"Teacher score parquet not found: {score_path}")

    score_df = pd.read_parquet(score_path)
    required = {"tokenized_index", score_field}
    missing = required.difference(score_df.columns)
    if missing:
        raise ValueError(f"Teacher score parquet missing columns: {sorted(missing)}")

    order = np.asarray(score_df["tokenized_index"].to_numpy(), dtype=np.int64)
    scores = np.asarray(score_df[score_field].to_numpy(), dtype=np.float64)
    if order.size == 0:
        raise ValueError("Teacher score parquet is empty.")
    if np.any(order < 0):
        raise ValueError("Teacher score parquet contains negative tokenized_index.")

    out = np.zeros(order.max() + 1, dtype=np.float64)
    out[order] = scores
    return out


def run_teacher_downstream(
    teacher_run_dir: str,
    teacher_aux_run_dir: Optional[str],
    train_dataset,
    val_dataset,
    train_raw_dataset,
    save_dir: str,
    batch_size: int,
    retrain_epochs: int,
    lr: float,
    keep_ratio: float,
    phase5_strategy: str,
    phase5_score_norm: str,
    phase5_min_weight: float,
    phase5_weight_power: float,
    random_mode: str,
    random_seed: int,
    device: torch.device,
    score_field: str = "pred_prob_good",
    aux_score_field: str = "pred_teacher_ambiguity",
    dual_noise_end_frac: float = 0.35,
    dual_ambiguity_start_frac: float = 0.2,
    dual_ambiguity_peak_frac: float = 0.55,
    dual_ambiguity_end_frac: float = 0.85,
    dual_noise_strength: float = 1.0,
    dual_ambiguity_strength: float = 1.0,
):
    if phase5_strategy not in {"filter", "weighted_sampler", "curriculum_sampler", "dual_curriculum_sampler"}:
        raise ValueError(f"Unsupported phase5_strategy for teacher downstream: {phase5_strategy}")

    teacher_results, audit_results = load_teacher_run_context(teacher_run_dir)
    scores = load_teacher_scores(teacher_run_dir, score_field=score_field)
    if len(scores) != len(train_dataset):
        raise ValueError(f"Teacher scores length {len(scores)} != train dataset length {len(train_dataset)}")
    aux_scores = None
    aux_teacher_results = None
    if phase5_strategy == "dual_curriculum_sampler":
        if not teacher_aux_run_dir:
            raise ValueError("dual_curriculum_sampler requires teacher_aux_run_dir.")
        aux_teacher_results, _ = load_teacher_run_context(teacher_aux_run_dir)
        _assert_matching_resolved_config(teacher_results, aux_teacher_results)
        aux_scores = load_teacher_scores(teacher_aux_run_dir, score_field=aux_score_field)
        if len(aux_scores) != len(train_dataset):
            raise ValueError(
                f"Aux teacher scores length {len(aux_scores)} != train dataset length {len(train_dataset)}"
            )

    os.makedirs(save_dir, exist_ok=True)
    baseline_ref = audit_results.get("baseline", {})
    results = {
        "teacher_run_dir": teacher_run_dir,
        "teacher_aux_run_dir": teacher_aux_run_dir,
        "teacher_audit_run_dir": teacher_results.get("teacher_audit_run_dir"),
        "baseline_ref": baseline_ref,
        "score_field": score_field,
        "aux_score_field": aux_score_field if teacher_aux_run_dir else None,
        "phase5_strategy": phase5_strategy,
    }

    if phase5_strategy == "filter":
        keep_act = max(1, min(len(train_dataset), int(round(float(keep_ratio) * len(train_dataset)))))
        kept_indices = np.argsort(scores)[::-1][:keep_act].astype(np.int64)
        kept_indices = np.sort(kept_indices)
        filtered_dataset = train_dataset.select(kept_indices.tolist())
        filtered_loader, val_loader = build_dataloaders(filtered_dataset, val_dataset, batch_size=batch_size)
        retrained_result = train_baseline(
            train_loader=filtered_loader,
            val_loader=val_loader,
            epochs=retrain_epochs,
            lr=lr,
            save_dir=os.path.join(save_dir, "phase5_retrained_teacher"),
            tag="teacher_filter",
            device=device,
        )

        sources = _extract_sources_for_tokenized(train_dataset, train_raw_dataset)
        random_indices, random_info = _sample_random_indices(
            mode=random_mode,
            keep_act=keep_act,
            keep_ratio=float(keep_ratio),
            sources=sources,
            kept_indices=kept_indices,
            seed=int(random_seed),
        )
        random_dataset = train_dataset.select(random_indices.tolist())
        random_loader, val_loader = build_dataloaders(random_dataset, val_dataset, batch_size=batch_size)
        random_result = train_baseline(
            train_loader=random_loader,
            val_loader=val_loader,
            epochs=retrain_epochs,
            lr=lr,
            save_dir=os.path.join(save_dir, "phase5_random_teacher"),
            tag="teacher_random",
            device=device,
        )
        results["teacher_retrained"] = {
            "metrics": retrained_result["best_metrics"],
            "history": retrained_result["history"],
            "keep_act": int(keep_act),
            "keep_ratio": float(keep_ratio),
        }
        results["teacher_random"] = {
            "mode": random_mode,
            "seed": int(random_seed),
            "info": random_info,
            "metrics": random_result["best_metrics"],
        }
    else:
        sources = _extract_sources_for_tokenized(train_dataset, train_raw_dataset)
        retrain_loader_factory, policy_info = build_phase5_policy(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            raw_train_dataset=train_raw_dataset,
            scores=scores,
            batch_size=batch_size,
            strategy=phase5_strategy,
            score_norm=phase5_score_norm,
            min_weight=phase5_min_weight,
            weight_power=phase5_weight_power,
            aux_scores=aux_scores,
            aux_score_norm=phase5_score_norm,
            dual_noise_end_frac=dual_noise_end_frac,
            dual_ambiguity_start_frac=dual_ambiguity_start_frac,
            dual_ambiguity_peak_frac=dual_ambiguity_peak_frac,
            dual_ambiguity_end_frac=dual_ambiguity_end_frac,
            dual_noise_strength=dual_noise_strength,
            dual_ambiguity_strength=dual_ambiguity_strength,
        )
        retrain_loader = retrain_loader_factory(1, retrain_epochs)
        _, val_loader = build_dataloaders(train_dataset, val_dataset, batch_size=batch_size)
        retrained_result = train_baseline(
            train_loader=retrain_loader,
            val_loader=val_loader,
            epochs=retrain_epochs,
            lr=lr,
            save_dir=os.path.join(save_dir, "phase5_retrained_teacher"),
            tag="teacher_policy",
            device=device,
            train_loader_factory=retrain_loader_factory,
        )
        results["teacher_retrained"] = {
            "metrics": retrained_result["best_metrics"],
            "history": retrained_result["history"],
            "policy": policy_info,
        }
        random_scores, random_aux_scores, random_info = _randomize_score_bundle_for_policy(
            primary_scores=scores,
            aux_scores=aux_scores,
            sources=sources,
            score_norm=phase5_score_norm,
            mode=random_mode,
            seed=int(random_seed),
        )
        if random_scores is None:
            results["teacher_random"] = {
                "mode": random_mode,
                "seed": int(random_seed),
                "info": random_info,
                "metrics": baseline_ref.get("best_metrics", {}),
                "source": "audit_baseline_ref",
            }
        else:
            random_loader_factory, random_policy_info = build_phase5_policy(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                raw_train_dataset=train_raw_dataset,
                scores=random_scores,
                batch_size=batch_size,
                strategy=phase5_strategy,
                score_norm=phase5_score_norm,
                min_weight=phase5_min_weight,
                weight_power=phase5_weight_power,
                aux_scores=random_aux_scores,
                aux_score_norm=phase5_score_norm,
                dual_noise_end_frac=dual_noise_end_frac,
                dual_ambiguity_start_frac=dual_ambiguity_start_frac,
                dual_ambiguity_peak_frac=dual_ambiguity_peak_frac,
                dual_ambiguity_end_frac=dual_ambiguity_end_frac,
                dual_noise_strength=dual_noise_strength,
                dual_ambiguity_strength=dual_ambiguity_strength,
            )
            random_loader = random_loader_factory(1, retrain_epochs)
            _, val_loader = build_dataloaders(train_dataset, val_dataset, batch_size=batch_size)
            random_result = train_baseline(
                train_loader=random_loader,
                val_loader=val_loader,
                epochs=retrain_epochs,
                lr=lr,
                save_dir=os.path.join(save_dir, "phase5_random_teacher"),
                tag="teacher_random_policy",
                device=device,
                train_loader_factory=random_loader_factory,
            )
            results["teacher_random"] = {
                "mode": random_mode,
                "seed": int(random_seed),
                "info": random_info,
                "metrics": random_result["best_metrics"],
                "policy": random_policy_info,
            }

    with open(os.path.join(save_dir, "teacher_downstream_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info("[teacher-downstream] Saved results -> %s", os.path.join(save_dir, "teacher_downstream_results.json"))
    return results
