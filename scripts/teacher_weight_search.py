#!/usr/bin/env python3
"""
teacher_weight_search.py

Offline search over teacher-formula weights using an existing sample-audit run.

Workflow per candidate:
1. Rebuild teacher columns from an existing audit cache.
2. Train a supervised teacher against the rebuilt target.
3. Rank candidates by teacher-only validation metrics.

This is intentionally cheaper than running downstream strict evaluation for
every candidate. We use it as a first-pass filter, then downstream only on the
best few formulas.
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_utils import prepare_data
from sample_audit import DEFAULT_TEACHER_FORMULAS, rebuild_teacher_cache
from teacher_trainer import train_teacher_datarater


LOG = logging.getLogger("teacher_weight_search")


TARGET_FIELD_BY_SIGNAL = {
    "badness": "teacher_badness_rank",
    "noise_risk": "teacher_noise_risk_rank",
    "ambiguity": "teacher_ambiguity_rank",
}


DEFAULT_FEATURES_BY_SIGNAL = {
    "badness": [
        "rank_mean_sq_error",
        "rank_late_mean_sq_error",
        "rank_final_sq_error",
        "rank_std_sq_error",
        "rank_sq_error_volatility",
        "one_minus_rank_improvement_ratio",
    ],
    "noise_risk": [
        "rank_mean_sq_error",
        "rank_late_mean_sq_error",
        "rank_final_sq_error",
        "rank_std_sq_error",
        "rank_sq_error_volatility",
        "rank_prediction_std",
        "one_minus_rank_improvement_ratio",
    ],
    "ambiguity": [
        "rank_prediction_std",
        "rank_sq_error_volatility",
        "mid_difficulty",
        "rank_improvement_ratio",
        "rank_late_mean_sq_error",
        "rank_mean_sq_error",
    ],
}


def parse_args():
    p = argparse.ArgumentParser(description="Offline teacher-formula weight search")
    p.add_argument("--source_audit_run_dir", required=True, help="Existing audit run directory")
    p.add_argument("--output_dir", required=True, help="Search output directory")
    p.add_argument("--signal", required=True, choices=["badness", "noise_risk", "ambiguity"])
    p.add_argument("--features", type=str, default=None,
                   help="Comma-separated feature names to search over. Defaults are signal-specific.")
    p.add_argument("--num_candidates", type=int, default=8,
                   help="Total candidates including the default formula")
    p.add_argument("--teacher_arch", type=str, default="multihead", choices=["single", "multihead", "moe"])
    p.add_argument("--teacher_epochs", type=int, default=8)
    p.add_argument("--teacher_lr", type=float, default=1e-4)
    p.add_argument("--teacher_val_frac", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=None,
                   help="Optional override. Defaults to the audit run config.")
    p.add_argument("--seed", type=int, default=None,
                   help="Optional override. Defaults to the audit run config.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--top_frac", type=float, default=None,
                   help="Optional teacher label fraction override when rebuilding the cache.")
    p.add_argument("--initial_spec", type=str, default=None,
                   help="Optional extra candidate formula, e.g. feat1=0.5,feat2=0.5")
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--router_top_k", type=int, default=2)
    p.add_argument("--capacity_factor", type=float, default=1.25)
    p.add_argument("--router_temperature", type=float, default=1.0)
    p.add_argument("--router_noise_std", type=float, default=0.0)
    p.add_argument("--moe_score_merge", type=str, default="weighted_sum", choices=["weighted_sum", "top1_only"])
    p.add_argument("--drop_overflow_tokens", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def setup_logging(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "search.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
        force=True,
    )


def _normalize_weights(weight_map: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(max(0.0, float(v)) for v in weight_map.values()))
    if total <= 0.0:
        raise ValueError("Formula weights must sum to a positive value.")
    return {k: float(v) / total for k, v in weight_map.items() if float(v) > 0.0}


def _spec_from_weights(weight_map: Dict[str, float]) -> str:
    items = sorted(weight_map.items())
    return ",".join(f"{k}={v:.8f}" for k, v in items)


def _parse_feature_list(raw: str, signal: str) -> List[str]:
    if raw is None or raw.strip() == "":
        return DEFAULT_FEATURES_BY_SIGNAL[signal]
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_spec(raw: str) -> Dict[str, float]:
    out = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        key, val = part.split("=", 1)
        out[key.strip()] = float(val.strip())
    return _normalize_weights(out)


def _sample_candidate_specs(signal: str, features: List[str], num_candidates: int, seed: int, initial_spec: str):
    rng = np.random.default_rng(seed)
    specs: List[Dict[str, object]] = []

    specs.append({
        "candidate_id": "default",
        "weights": _normalize_weights(DEFAULT_TEACHER_FORMULAS[signal]),
        "source": "default_formula",
    })

    uniform = _normalize_weights({feature: 1.0 for feature in features})
    specs.append({
        "candidate_id": "uniform",
        "weights": uniform,
        "source": "uniform_over_features",
    })

    if initial_spec:
        specs.append({
            "candidate_id": "initial",
            "weights": _parse_spec(initial_spec),
            "source": "user_initial_spec",
        })

    idx = 0
    while len(specs) < num_candidates:
        sampled = rng.dirichlet(np.ones(len(features), dtype=np.float32))
        weights = _normalize_weights({feature: float(weight) for feature, weight in zip(features, sampled.tolist())})
        if any(existing["weights"] == weights for existing in specs):
            continue
        specs.append({
            "candidate_id": f"rand_{idx:03d}",
            "weights": weights,
            "source": "dirichlet",
        })
        idx += 1
    return specs[:num_candidates]


def _load_audit_config(source_audit_run_dir: str) -> Dict[str, object]:
    cfg_path = os.path.join(source_audit_run_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing audit config: {cfg_path}")
    return json.load(open(cfg_path))


def _sanitize(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"search_{args.signal}_{timestamp}")
    setup_logging(run_dir)
    LOG.info("Teacher weight search starting | signal=%s | run_dir=%s", args.signal, run_dir)
    LOG.info("Args: %s", json.dumps(vars(args), indent=2))

    cfg = _load_audit_config(args.source_audit_run_dir)
    dataset_name = str(cfg.get("dataset", "Bindwell/PPBA"))
    data_mode = str(cfg.get("data_mode", "all"))
    train_ratio = float(cfg.get("train_ratio", 0.8))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    max_length = int(cfg.get("max_length", 512))
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.get("batch_size", 32))
    top_frac = float(args.top_frac if args.top_frac is not None else cfg.get("audit_top_frac", 0.2))
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    features = _parse_feature_list(args.features, signal=args.signal)
    target_field = TARGET_FIELD_BY_SIGNAL[args.signal]

    with open(os.path.join(run_dir, "search_config.json"), "w") as f:
        json.dump({
            "args": vars(args),
            "resolved_data_config": {
                "dataset": dataset_name,
                "data_mode": data_mode,
                "train_ratio": train_ratio,
                "seed": seed,
                "max_length": max_length,
                "batch_size": batch_size,
                "top_frac": top_frac,
            },
            "features": features,
            "target_field": target_field,
        }, f, indent=2)

    LOG.info("Preparing tokenized train dataset once for all candidates.")
    _, _, train_dataset, _, train_raw_dataset, _ = prepare_data(
        dataset_name=dataset_name,
        max_length=max_length,
        batch_size=batch_size,
        train_ratio=train_ratio,
        seed=seed,
        mode=data_mode,
    )

    candidates = _sample_candidate_specs(
        signal=args.signal,
        features=features,
        num_candidates=args.num_candidates,
        seed=seed,
        initial_spec=args.initial_spec,
    )

    all_results = []
    for cand_idx, cand in enumerate(candidates, start=1):
        cand_id = str(cand["candidate_id"])
        cand_dir = os.path.join(run_dir, cand_id)
        rebuild_dir = os.path.join(cand_dir, "rebuild")
        teacher_dir = os.path.join(cand_dir, "teacher_datarater")
        os.makedirs(cand_dir, exist_ok=True)

        weight_spec = _spec_from_weights(cand["weights"])
        LOG.info("[%d/%d] Candidate %s | source=%s | spec=%s",
                 cand_idx, len(candidates), cand_id, cand["source"], weight_spec)

        results_path = os.path.join(cand_dir, "candidate_results.json")
        if os.path.exists(results_path):
            LOG.info("Candidate %s already finished. Loading cached result.", cand_id)
            with open(results_path) as f:
                all_results.append(json.load(f))
            continue

        rebuild_kwargs = {
            "source_audit_run_dir": args.source_audit_run_dir,
            "rebuilt_run_dir": rebuild_dir,
            "top_frac": top_frac,
            "badness_weight_spec": None,
            "noise_weight_spec": None,
            "ambiguity_weight_spec": None,
        }
        if args.signal == "badness":
            rebuild_kwargs["badness_weight_spec"] = weight_spec
        elif args.signal == "noise_risk":
            rebuild_kwargs["noise_weight_spec"] = weight_spec
        elif args.signal == "ambiguity":
            rebuild_kwargs["ambiguity_weight_spec"] = weight_spec
        else:
            raise ValueError(f"Unsupported signal: {args.signal}")

        rebuild_result = rebuild_teacher_cache(**rebuild_kwargs)

        teacher_result = train_teacher_datarater(
            train_dataset=train_dataset,
            raw_train_dataset=train_raw_dataset,
            audit_parquet_path=rebuild_result["sample_audit_parquet"],
            save_dir=teacher_dir,
            teacher_arch=args.teacher_arch,
            epochs=args.teacher_epochs,
            batch_size=batch_size,
            lr=args.teacher_lr,
            val_frac=args.teacher_val_frac,
            seed=seed,
            device=device,
            num_experts=args.num_experts,
            router_top_k=args.router_top_k,
            capacity_factor=args.capacity_factor,
            router_temperature=args.router_temperature,
            router_noise_std=args.router_noise_std,
            moe_score_merge=args.moe_score_merge,
            drop_overflow_tokens=args.drop_overflow_tokens,
            target_mode="regression_all",
            regression_field=target_field,
        )

        metrics = teacher_result["best_metrics"]
        candidate_result = {
            "candidate_id": cand_id,
            "candidate_index": cand_idx,
            "signal": args.signal,
            "feature_source": cand["source"],
            "weights": cand["weights"],
            "weight_spec": weight_spec,
            "target_field": target_field,
            "metrics": metrics,
            "teacher_dir": teacher_dir,
            "rebuild_dir": rebuild_dir,
        }
        with open(results_path, "w") as f:
            json.dump(_sanitize(candidate_result), f, indent=2)
        all_results.append(_sanitize(candidate_result))

    all_results.sort(key=lambda row: (-float(row["metrics"].get("spearman_target", 0.0)),
                                      float(row["metrics"].get("mse", 1e9))))

    summary = {
        "signal": args.signal,
        "run_dir": run_dir,
        "target_field": target_field,
        "features": features,
        "num_candidates": len(all_results),
        "best_candidate": all_results[0] if all_results else None,
        "candidates": all_results,
    }
    with open(os.path.join(run_dir, "search_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(run_dir, "search_results.csv"), "w") as f:
        header = [
            "rank", "candidate_id", "feature_source", "weight_spec",
            "spearman_target", "pearson_target", "mse", "mae", "extreme_auc"
        ]
        f.write(",".join(header) + "\n")
        for rank, row in enumerate(all_results, start=1):
            m = row["metrics"]
            vals = [
                rank,
                row["candidate_id"],
                row["feature_source"],
                row["weight_spec"],
                m.get("spearman_target", ""),
                m.get("pearson_target", ""),
                m.get("mse", ""),
                m.get("mae", ""),
                m.get("extreme_auc", ""),
            ]
            f.write(",".join(str(v) for v in vals) + "\n")

    if all_results:
        best = all_results[0]
        LOG.info(
            "Best candidate %s | spearman=%.4f | pearson=%.4f | mse=%.4f | spec=%s",
            best["candidate_id"],
            float(best["metrics"].get("spearman_target", 0.0)),
            float(best["metrics"].get("pearson_target", 0.0)),
            float(best["metrics"].get("mse", 0.0)),
            best["weight_spec"],
        )
    LOG.info("Teacher weight search complete -> %s", os.path.join(run_dir, "search_results.json"))


if __name__ == "__main__":
    main()
