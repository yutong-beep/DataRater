#!/usr/bin/env python3
"""
main.py — Full DataRater Pipeline Entry Point
================================================

Runs all 5 phases sequentially:
  Phase 1: Baseline training 
  Phase 2: Meta-train DataRater (Uses specialized meta_batch_size to prevent OOM)
  Phase 3: Score training data, build CDF
  Phase 4: Filter dataset using P_accept
  Phase 5: Retrain on curated data & compare

Usage:
    python main.py --epochs 10 --meta_steps 5000 --retrain_epochs 10 --batch_size 64 --meta_batch_size 16 --B 64
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from collections import Counter

def _cli_flag_present(flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:])


if _cli_flag_present("--strict_deterministic"):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import numpy as np

# ==========================================
# Argument Parsing
# ==========================================
def parse_args():
    p = argparse.ArgumentParser(description="DataRater Full Pipeline")

    # General
    p.add_argument("--phase", type=str, default="1,2,3,4,5",
                   help="Comma-separated phases to run (e.g. '1,2,3,4,5' or '1')")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strict_deterministic", action="store_true",
                   help="Enable stricter PyTorch/DataLoader determinism for reproducibility-focused reruns")
    p.add_argument("--num_workers", type=int, default=2,
                   help="DataLoader worker count (default: 2)")
    p.add_argument("--device", type=str, default=None,
                   help="Device (default: auto-detect)")
    p.add_argument("--output_dir", type=str, default="experiments",
                   help="Root output directory")
    p.add_argument("--random_only", action="store_true",
                   help="Run only random-baseline retrain for an existing run_dir, then exit")
    p.add_argument("--run_dir", type=str, default=None,
                   help="Existing run directory used by --random_only")
    p.add_argument("--teacher_train_only", action="store_true",
                   help="Train a supervised DataRater from a completed Phase-1 sample-audit run, then exit")
    p.add_argument("--teacher_audit_run_dir", type=str, default=None,
                   help="Existing audit run directory used by --teacher_train_only")
    p.add_argument("--teacher_rebuild_only", action="store_true",
                   help="Rebuild teacher columns from an existing sample-audit cache, then exit")
    p.add_argument("--teacher_downstream_only", action="store_true",
                   help="Use a completed supervised teacher run to drive downstream retraining, then exit")
    p.add_argument("--teacher_run_dir", type=str, default=None,
                   help="Existing supervised teacher run directory used by --teacher_downstream_only")
    p.add_argument("--teacher_aux_run_dir", type=str, default=None,
                   help="Optional second supervised teacher run directory used by dual-head downstream policies")

    # Data
    p.add_argument("--dataset", type=str, default="Bindwell/PPBA")
    p.add_argument("--data_mode", type=str, default="combined_train", choices=["combined_train", "all"],
                   help="Dataset loading mode: 'combined_train' (default) or 'all'")
    p.add_argument("--exclude_sources", type=str, default="",
                   help="Comma-separated source names to exclude before splitting/tokenization")
    p.add_argument("--max_length", type=int, default=512,
                   help="Max sequence length for tokenization")
    p.add_argument("--esm_attn_implementation", type=str, default="auto",
                   choices=["auto", "sdpa", "flash_attention_2", "eager"],
                   help="Transformers attention backend used when loading ESM")
    p.add_argument("--esm_torch_dtype", type=str, default="auto",
                   choices=["auto", "float32", "float16", "bfloat16"],
                   help="torch_dtype used when loading ESM from pretrained")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size for Baseline and Retrain (Phase 1 & 5)")
    p.add_argument("--meta_batch_size", type=int, default=16,
                   help="Smaller batch size specifically for DataRater Meta-Training to prevent OOM (Phase 2)")
    p.add_argument("--train_ratio", type=float, default=0.8)

    # Phase 1 & 5: Baseline training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--collect_sample_audit", action="store_true",
                   help="During Phase 1, run train-set eval after each epoch and export a teacher-signal cache")
    p.add_argument("--audit_top_frac", type=float, default=0.2,
                   help="Per-source top/bottom fraction used for teacher good/bad labels")
    p.add_argument("--teacher_arch", type=str, default="multihead",
                   choices=["single", "multihead", "moe"],
                   help="Architecture used by --teacher_train_only")
    p.add_argument("--teacher_epochs", type=int, default=10,
                   help="Epochs used by --teacher_train_only")
    p.add_argument("--teacher_lr", type=float, default=1e-4,
                   help="Learning rate used by --teacher_train_only")
    p.add_argument("--teacher_val_frac", type=float, default=0.2,
                   help="Holdout fraction for supervised teacher training")
    p.add_argument("--teacher_target_mode", type=str, default="binary_extremes",
                   choices=["binary_extremes", "regression_all"],
                   help="Supervised teacher target: top/bottom binary labels or full-dataset continuous regression")
    p.add_argument("--teacher_regression_field", type=str, default="teacher_goodness",
                   choices=[
                       "teacher_goodness",
                       "teacher_badness_rank",
                       "teacher_badness",
                       "teacher_noise_risk",
                       "teacher_noise_risk_rank",
                       "teacher_ambiguity",
                       "teacher_ambiguity_rank",
                   ],
                   help="Continuous teacher target used when --teacher_target_mode regression_all")
    p.add_argument("--teacher_score_field", type=str, default="pred_prob_good",
                   choices=[
                       "pred_prob_good",
                       "pred_logit_good",
                       "pred_teacher_goodness",
                       "pred_teacher_badness",
                       "pred_teacher_noise_risk",
                       "pred_teacher_ambiguity",
                       "pred_sigmoid_score",
                       "pred_raw_score",
                   ],
                   help="Teacher score column used by --teacher_downstream_only")
    p.add_argument("--teacher_aux_score_field", type=str, default="pred_teacher_ambiguity",
                   choices=[
                       "pred_prob_good",
                       "pred_logit_good",
                       "pred_teacher_goodness",
                       "pred_teacher_badness",
                       "pred_teacher_noise_risk",
                       "pred_teacher_ambiguity",
                       "pred_sigmoid_score",
                       "pred_raw_score",
                   ],
                   help="Auxiliary teacher score column used by dual-head downstream policies")
    p.add_argument("--teacher_badness_weights", type=str, default=None,
                   help="Comma-separated feature=weight spec used to rebuild teacher_badness from audit features")
    p.add_argument("--teacher_noise_weights", type=str, default=None,
                   help="Comma-separated feature=weight spec used to rebuild teacher_noise_risk from audit features")
    p.add_argument("--teacher_ambiguity_weights", type=str, default=None,
                   help="Comma-separated feature=weight spec used to rebuild teacher_ambiguity from audit features")
    p.add_argument("--teacher_rebuild_top_frac", type=float, default=None,
                   help="Optional top fraction override when rebuilding teacher labels from an existing audit cache")

    # Phase 2: Meta-training
    p.add_argument("--meta_steps", type=int, default=5000,
                   help="Number of meta-training steps")
    p.add_argument("--n_inner_models", type=int, default=8)
    p.add_argument("--lifetime", type=int, default=2000,
                   help="Inner model lifetime before re-init")
    p.add_argument("--T_window", type=int, default=2,
                   help="Truncated inner loop window (paired with --T_backprop for v2 truncated BPTT)")
    # v2 additions
    p.add_argument("--T_backprop", type=int, default=2,
                   help="Number of inner steps to backprop through (rest are detached warmup). "
                        "Set < T_window to save memory with large T_window.")
    p.add_argument("--use_zscore_inner", action="store_true",
                   help="v2: Per-source z-score normalize inner targets to equalize source difficulty")
    p.add_argument("--temperature", type=float, default=0.5,
                   help="Softmax temperature tau used by DataRater in meta-training")
    p.add_argument("--outer_objective", type=str, default="mse_norm",
                   choices=["mse", "rmse", "mse_norm", "pearson", "cosine", "mix", "source_stratified_mse"],
                   help="Outer-loop objective for meta-training")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Mix coefficient for outer_objective='mix'")
    p.add_argument("--outer_eps", type=float, default=1e-8,
                   help="Numerical epsilon for pearson/cosine outer objectives")
    p.add_argument("--mse_norm_eps", type=float, default=1e-6,
                   help="Numerical epsilon for source-normalized MSE outer objective")
    p.add_argument("--ablation", action="store_true",
                   help="Use first-order ablation (Task c)")
    p.add_argument("--sample_one_inner", action="store_true",
                   help="Sample 1 inner model per meta-step (Task f)")
    p.add_argument("--meta_grad_clip", type=float, default=None,
                   help="Optional max norm for DataRater meta-gradient clipping")
    p.add_argument("--canary_interval", type=int, default=0,
                   help="Run source-spread canary diagnostics every N meta-steps (0 disables)")
    p.add_argument("--inner_reset_strategy", type=str, default="random_init",
                   choices=["random_init", "carryover", "checkpoint_bank"],
                   help="How inner models are refreshed during meta-training")
    p.add_argument("--inner_init_bank_dir", type=str, default=None,
                   help="Directory containing pretrained inner-model checkpoints used by checkpoint-bank resets")
    p.add_argument("--inner_init_bank_jitter", type=int, default=0,
                   help="Neighbor radius when sampling from the inner-init checkpoint bank on reset")
    p.add_argument("--datarater_arch", type=str, default="single",
                   choices=["single", "multihead", "moe"],
                   help="DataRater architecture for Phase 2")
    p.add_argument("--outer_sampling", type=str, default="random",
                   choices=["random", "balanced", "harder", "dataset_ratio", "custom_ratio"],
                   help="Outer-batch sampling mode for Phase 2")
    p.add_argument("--outer_per_source", type=int, default=None,
                   help="Samples per source in each balanced/harder outer batch (default: auto)")
    p.add_argument("--hard_outer_sources", type=str, default="SKEMPI v2.0,PDBbind v2020",
                   help="Comma-separated source names used when --outer_sampling harder")
    p.add_argument("--outer_source_weights", type=str, default="",
                   help="Comma-separated source=weight spec used when --outer_sampling custom_ratio")
    p.add_argument("--inner_batch_scope", type=str, default="shared",
                   choices=["shared", "per_inner"],
                   help="Whether inner models share the same inner batches within a meta-step")
    p.add_argument("--outer_batch_scope", type=str, default="shared",
                   choices=["shared", "per_inner"],
                   help="Whether inner models share the same outer batch within a meta-step")
    p.add_argument("--num_experts", type=int, default=4,
                   help="Number of experts when --datarater_arch moe")
    p.add_argument("--router_top_k", type=int, default=2, choices=[1, 2],
                   help="Sparse router top-k for MoE DataRater")
    p.add_argument("--capacity_factor", type=float, default=1.25,
                   help="Capacity multiplier used by MoE expert dispatch")
    p.add_argument("--router_aux_loss_coef", type=float, default=0.01,
                   help="Coefficient for MoE router load-balancing auxiliary loss")
    p.add_argument("--router_z_loss_coef", type=float, default=0.0,
                   help="Coefficient for MoE router z-loss regularization")
    p.add_argument("--router_noise_std", type=float, default=0.0,
                   help="Gaussian jitter std added to MoE router logits during training")
    p.add_argument("--router_temperature", type=float, default=1.0,
                   help="Temperature applied to MoE router logits before softmax")
    p.add_argument("--moe_score_merge", type=str, default="weighted_sum",
                   choices=["weighted_sum", "top1_only"],
                   help="How to merge MoE expert scores back to a scalar")
    p.add_argument("--drop_overflow_tokens", action=argparse.BooleanOptionalAction, default=True,
                   help="Whether MoE should drop overflowed expert assignments")

    # Phase 3-4: Scoring & filtering
    p.add_argument("--N_ref", type=int, default=10000,
                   help="Number of reference points for CDF")
    p.add_argument("--B", type=int, default=64,
                   help="Batch size B used strictly for P_accept formula calculation")
    p.add_argument("--keep_ratio", type=float, default=0.7,
                   help="Target keep ratio for filtering")

    # Phase 5: Retrain
    p.add_argument("--retrain_epochs", type=int, default=10,
                   help="Epochs for retraining on filtered data")
    p.add_argument("--phase5_strategy", type=str, default="filter",
                   choices=["filter", "weighted_sampler", "curriculum_sampler", "dual_curriculum_sampler"],
                   help="Phase 5 retraining policy: hard-filter subset (default) or full-data score-driven sampling")
    p.add_argument("--phase5_score_norm", type=str, default="source_rank",
                   choices=["source_rank", "global_rank"],
                   help="How frozen DataRater scores are normalized before Phase 5 score-driven sampling")
    p.add_argument("--phase5_min_weight", type=float, default=0.1,
                   help="Minimum per-sample weight added before Phase 5 weighted sampling")
    p.add_argument("--phase5_weight_power", type=float, default=2.0,
                   help="Exponent applied to normalized DataRater quality before Phase 5 weighted sampling")
    p.add_argument("--phase5_dual_noise_end_frac", type=float, default=0.35,
                   help="Fraction of training over which the dual-head policy suppresses high-noise samples")
    p.add_argument("--phase5_dual_ambiguity_start_frac", type=float, default=0.2,
                   help="Training fraction where ambiguity upweighting starts in dual-head curriculum")
    p.add_argument("--phase5_dual_ambiguity_peak_frac", type=float, default=0.55,
                   help="Training fraction where ambiguity upweighting peaks in dual-head curriculum")
    p.add_argument("--phase5_dual_ambiguity_end_frac", type=float, default=0.85,
                   help="Training fraction where ambiguity upweighting returns to zero")
    p.add_argument("--phase5_dual_noise_strength", type=float, default=1.0,
                   help="Multiplier on the early noise-suppression branch of dual-head curriculum")
    p.add_argument("--phase5_dual_ambiguity_strength", type=float, default=1.0,
                   help="Multiplier on the mid-training ambiguity branch of dual-head curriculum")
    p.add_argument("--random_baseline", action="store_true",
                   help="Also run matched random baseline retrain in Phase 5")
    p.add_argument("--random_seed", type=int, default=42,
                   help="Random seed for random baseline subset sampling")
    p.add_argument("--random_mode", type=str, default="matched_source_counts",
                   choices=["matched_source_counts", "stratified_ratio", "uniform"],
                   help="Random baseline subset strategy")

    # Checkpoints (for resuming)
    p.add_argument("--datarater_ckpt", type=str, default=None,
                   help="Path to pre-trained DataRater checkpoint (skip Phase 2)")

    return p.parse_args()


def _flag_set(flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:])


def _resolve_param(name: str, cli_value, saved_cfg: dict, default):
    flag = f"--{name}"
    if _flag_set(flag):
        return cli_value
    return saved_cfg.get(name, default)


def _parse_csv_arg(csv_arg: str) -> list:
    if csv_arg is None:
        return []
    return [part.strip() for part in str(csv_arg).split(",") if part.strip()]


def _parse_weight_spec(spec: str) -> dict:
    out = {}
    if spec is None:
        return out
    for part in str(spec).split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid weight spec item '{item}'. Expected source=weight.")
        key, value = item.split("=", 1)
        out[key.strip()] = float(value.strip())
    return out


def _configure_reproducibility(seed: int, strict: bool = False):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if strict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False


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

    if keep_act > n:
        raise ValueError(f"keep_act={keep_act} cannot exceed dataset size={n}")

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
        if len(out) != keep_act:
            raise ValueError(f"Matched source counts produced {len(out)} != keep_act {keep_act}")
        return out, {"mode": mode, "counts_kept": dict(counts_kept)}

    # stratified_ratio
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


def run_random_only(args):
    if not args.run_dir:
        raise ValueError("--run_dir is required when --random_only is set.")
    run_dir = args.run_dir
    cfg_path = os.path.join(run_dir, "config.json")
    results_path = os.path.join(run_dir, "results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"results.json not found: {results_path}")
    saved_cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    existing_results = json.load(open(results_path))

    from data_utils import prepare_data
    from baseline_trainer import train_baseline

    dataset_name = _resolve_param("dataset", args.dataset, saved_cfg, "Bindwell/PPBA")
    data_mode = _resolve_param("data_mode", args.data_mode, saved_cfg, "combined_train")
    exclude_sources = _parse_csv_arg(_resolve_param("exclude_sources", args.exclude_sources, saved_cfg, ""))
    train_ratio = _resolve_param("train_ratio", args.train_ratio, saved_cfg, 0.8)
    seed = _resolve_param("seed", args.seed, saved_cfg, 42)
    max_length = _resolve_param("max_length", args.max_length, saved_cfg, 512)
    batch_size = _resolve_param("batch_size", args.batch_size, saved_cfg, 64)
    lr = _resolve_param("lr", args.lr, saved_cfg, 1e-4)
    retrain_epochs = _resolve_param("retrain_epochs", args.retrain_epochs, saved_cfg, 10)

    train_loader, val_loader, train_tok, val_tok, train_raw, val_raw = prepare_data(
        dataset_name=dataset_name,
        max_length=max_length,
        batch_size=batch_size,
        train_ratio=train_ratio,
        seed=seed,
        mode=data_mode,
        exclude_sources=exclude_sources,
    )

    keep_ratio = float(existing_results.get("filtering", {}).get("target_keep_ratio", args.keep_ratio))
    kept_indices_path = os.path.join(run_dir, "phase34_scoring", "kept_indices.npy")
    mode_used = args.random_mode
    if os.path.exists(kept_indices_path):
        kept_indices = np.load(kept_indices_path)
        keep_act = int(len(kept_indices))
    else:
        n_train = len(train_tok)
        keep_act = int(round(keep_ratio * n_train))
        keep_act = max(1, min(keep_act, n_train))
        kept_indices = np.array([], dtype=np.int64)
        if mode_used == "matched_source_counts":
            print(
                f"[WARN] kept indices not found: {kept_indices_path}. "
                "Fallback random_mode from 'matched_source_counts' to 'stratified_ratio'."
            )
            mode_used = "stratified_ratio"

    sources = _extract_sources_for_tokenized(train_tok, train_raw)
    random_indices, extra_info = _sample_random_indices(
        mode=mode_used,
        keep_act=keep_act,
        keep_ratio=keep_ratio,
        sources=sources,
        kept_indices=kept_indices,
        seed=args.random_seed,
    )

    run_tag = f"{mode_used}_seed{int(args.random_seed)}"
    phase5_random_dir = os.path.join(run_dir, f"phase5_random_{run_tag}")
    os.makedirs(phase5_random_dir, exist_ok=True)
    np.save(os.path.join(phase5_random_dir, "random_kept_indices.npy"), random_indices)
    random_info = {
        "mode": mode_used,
        "mode_requested": args.random_mode,
        "seed": int(args.random_seed),
        "keep_act": keep_act,
        "keep_ratio_target": keep_ratio,
        **extra_info,
    }
    with open(os.path.join(phase5_random_dir, "random_info.json"), "w") as f:
        json.dump(random_info, f, indent=2)

    random_filtered = train_tok.select(random_indices.tolist())
    random_result = train_baseline(
        train_loader=torch.utils.data.DataLoader(
            random_filtered, batch_size=batch_size, shuffle=True, num_workers=2,
            collate_fn=lambda b: {
                "input_ids": torch.stack([x["input_ids"] for x in b]),
                "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                "affinity": torch.tensor([x["affinity"] for x in b], dtype=torch.float32),
            },
            drop_last=True, pin_memory=True
        ),
        val_loader=val_loader,
        epochs=retrain_epochs,
        lr=lr,
        save_dir=phase5_random_dir,
        tag="random",
        device=torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        attn_implementation=args.esm_attn_implementation,
        esm_torch_dtype=args.esm_torch_dtype,
    )

    random_block = {
        "mode": mode_used,
        "mode_requested": args.random_mode,
        "seed": int(args.random_seed),
        "keep_act": keep_act,
        "metrics": random_result["best_metrics"],
    }
    with open(os.path.join(phase5_random_dir, "random_results.json"), "w") as f:
        json.dump(random_block, f, indent=2)

    existing_results["retrained_random"] = random_block
    random_runs = existing_results.get("retrained_random_runs", {})
    if not isinstance(random_runs, dict):
        random_runs = {}
    random_runs[run_tag] = random_block
    existing_results["retrained_random_runs"] = random_runs
    with open(results_path, "w") as f:
        json.dump(existing_results, f, indent=2)
    print(f"Random-only retrain complete. Saved to: {phase5_random_dir}")
    print(f"Updated: {results_path}")


def run_teacher_train_only(args):
    if not args.teacher_audit_run_dir:
        raise ValueError("--teacher_audit_run_dir is required when --teacher_train_only is set.")

    audit_run_dir = args.teacher_audit_run_dir
    cfg_path = os.path.join(audit_run_dir, "config.json")
    saved_cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}

    dataset_name = _resolve_param("dataset", args.dataset, saved_cfg, "Bindwell/PPBA")
    data_mode = _resolve_param("data_mode", args.data_mode, saved_cfg, "combined_train")
    exclude_sources = _parse_csv_arg(_resolve_param("exclude_sources", args.exclude_sources, saved_cfg, ""))
    train_ratio = _resolve_param("train_ratio", args.train_ratio, saved_cfg, 0.8)
    seed = _resolve_param("seed", args.seed, saved_cfg, 42)
    max_length = _resolve_param("max_length", args.max_length, saved_cfg, 512)
    batch_size = _resolve_param("batch_size", args.batch_size, saved_cfg, 64)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"p6_teacher_{args.teacher_arch}"
    run_dir = os.path.join(args.output_dir, f"{run_prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logging(run_dir)
    logger.info("=" * 70)
    logger.info("DataRater Teacher-Supervision Pipeline")
    logger.info("=" * 70)
    logger.info(f"Audit run dir: {audit_run_dir}")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Args: {json.dumps(vars(args), indent=2)}")
    logger.info(
        "Resolved data config | dataset=%s | data_mode=%s | train_ratio=%.3f | seed=%d | max_length=%d | batch_size=%d",
        dataset_name,
        data_mode,
        float(train_ratio),
        int(seed),
        int(max_length),
        int(batch_size),
    )
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    _configure_reproducibility(seed, strict=args.strict_deterministic)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    from data_utils import prepare_data
    from teacher_trainer import train_teacher_datarater

    _, _, train_dataset, _, train_raw_dataset, _ = prepare_data(
        dataset_name=dataset_name,
        max_length=max_length,
        batch_size=batch_size,
        train_ratio=train_ratio,
        seed=seed,
        num_workers=args.num_workers,
        deterministic=args.strict_deterministic,
        mode=data_mode,
        exclude_sources=exclude_sources,
    )

    audit_parquet_path = os.path.join(
        audit_run_dir,
        "phase1_baseline",
        "sample_audit",
        "sample_audit.parquet",
    )
    teacher_dir = os.path.join(run_dir, "teacher_datarater")
    teacher_result = train_teacher_datarater(
        train_dataset=train_dataset,
        raw_train_dataset=train_raw_dataset,
        audit_parquet_path=audit_parquet_path,
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
        target_mode=args.teacher_target_mode,
        regression_field=args.teacher_regression_field,
        attn_implementation=args.esm_attn_implementation,
        esm_torch_dtype=args.esm_torch_dtype,
    )

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

    results = {
        "teacher_audit_run_dir": audit_run_dir,
        "resolved_config": {
            "dataset": dataset_name,
            "data_mode": data_mode,
            "train_ratio": float(train_ratio),
            "seed": int(seed),
            "max_length": int(max_length),
            "batch_size": int(batch_size),
            "exclude_sources": exclude_sources,
        },
        "teacher_config": {
            "teacher_target_mode": args.teacher_target_mode,
            "teacher_regression_field": args.teacher_regression_field,
            "teacher_arch": args.teacher_arch,
            "teacher_epochs": int(args.teacher_epochs),
            "teacher_lr": float(args.teacher_lr),
            "teacher_val_frac": float(args.teacher_val_frac),
        },
        "teacher_supervised": teacher_result,
    }
    results_clean = _sanitize(results)
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_clean, f, indent=2)

    logger.info(f"All results saved -> {results_path}")
    logger.info(f"Run directory: {run_dir}")
    logger.info("Teacher-supervision pipeline complete!")


def run_teacher_rebuild_only(args):
    if not args.teacher_audit_run_dir:
        raise ValueError("--teacher_audit_run_dir is required when --teacher_rebuild_only is set.")

    source_audit_run_dir = args.teacher_audit_run_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = "p7_teacher_rebuild"
    run_dir = os.path.join(args.output_dir, f"{run_prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logging(run_dir)
    logger.info("=" * 70)
    logger.info("DataRater Teacher-Rebuild Pipeline")
    logger.info("=" * 70)
    logger.info(f"Source audit run dir: {source_audit_run_dir}")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Args: {json.dumps(vars(args), indent=2)}")
    with open(os.path.join(run_dir, "rebuild_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    from sample_audit import rebuild_teacher_cache

    rebuild_result = rebuild_teacher_cache(
        source_audit_run_dir=source_audit_run_dir,
        rebuilt_run_dir=run_dir,
        top_frac=args.teacher_rebuild_top_frac,
        badness_weight_spec=args.teacher_badness_weights,
        noise_weight_spec=args.teacher_noise_weights,
        ambiguity_weight_spec=args.teacher_ambiguity_weights,
    )

    logger.info("Rebuilt teacher cache -> %s", rebuild_result["sample_audit_parquet"])
    logger.info("All results saved -> %s", os.path.join(run_dir, "results.json"))
    logger.info("Teacher-rebuild pipeline complete!")


def run_teacher_downstream_only(args):
    if not args.teacher_run_dir:
        raise ValueError("--teacher_run_dir is required when --teacher_downstream_only is set.")
    if args.phase5_strategy == "dual_curriculum_sampler" and not args.teacher_aux_run_dir:
        raise ValueError("--teacher_aux_run_dir is required when --phase5_strategy dual_curriculum_sampler.")

    teacher_results_path = os.path.join(args.teacher_run_dir, "results.json")
    if not os.path.exists(teacher_results_path):
        raise FileNotFoundError(f"Teacher run results not found: {teacher_results_path}")
    teacher_results = json.load(open(teacher_results_path))
    resolved = teacher_results.get("resolved_config", {})
    if not resolved:
        raise ValueError(f"Missing resolved_config in teacher run results: {teacher_results_path}")

    if args.teacher_aux_run_dir:
        aux_results_path = os.path.join(args.teacher_aux_run_dir, "results.json")
        if not os.path.exists(aux_results_path):
            raise FileNotFoundError(f"Aux teacher run results not found: {aux_results_path}")
        aux_results = json.load(open(aux_results_path))
        aux_resolved = aux_results.get("resolved_config", {})
        keys = ["dataset", "data_mode", "train_ratio", "seed", "max_length", "batch_size"]
        mismatches = [k for k in keys if resolved.get(k) != aux_resolved.get(k)]
        if mismatches:
            raise ValueError(
                "Primary/aux teacher resolved_config mismatch: "
                + ", ".join(f"{k}={resolved.get(k)} vs {aux_resolved.get(k)}" for k in mismatches)
            )

    dataset_name = str(resolved["dataset"])
    data_mode = str(resolved["data_mode"])
    exclude_sources = [str(x) for x in resolved.get("exclude_sources", [])]
    train_ratio = float(resolved["train_ratio"])
    seed = int(resolved["seed"])
    max_length = int(resolved["max_length"])
    batch_size = int(resolved["batch_size"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"p6_teacher_downstream_{args.phase5_strategy}"
    run_dir = os.path.join(args.output_dir, f"{run_prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logging(run_dir)
    logger.info("=" * 70)
    logger.info("DataRater Teacher-Score Downstream Pipeline")
    logger.info("=" * 70)
    logger.info(f"Teacher run dir: {args.teacher_run_dir}")
    if args.teacher_aux_run_dir:
        logger.info(f"Teacher aux run dir: {args.teacher_aux_run_dir}")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Args: {json.dumps(vars(args), indent=2)}")
    logger.info(
        "Resolved data config | dataset=%s | data_mode=%s | train_ratio=%.3f | seed=%d | max_length=%d | batch_size=%d",
        dataset_name,
        data_mode,
        float(train_ratio),
        int(seed),
        int(max_length),
        int(batch_size),
    )
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    _configure_reproducibility(seed, strict=args.strict_deterministic)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    from data_utils import prepare_data
    from teacher_downstream import run_teacher_downstream

    _, val_loader, train_dataset, val_dataset, train_raw_dataset, _ = prepare_data(
        dataset_name=dataset_name,
        max_length=max_length,
        batch_size=batch_size,
        train_ratio=train_ratio,
        seed=seed,
        num_workers=args.num_workers,
        deterministic=args.strict_deterministic,
        mode=data_mode,
        exclude_sources=exclude_sources,
    )
    del val_loader

    downstream_result = run_teacher_downstream(
        teacher_run_dir=args.teacher_run_dir,
        teacher_aux_run_dir=args.teacher_aux_run_dir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_raw_dataset=train_raw_dataset,
        save_dir=os.path.join(run_dir, "teacher_downstream"),
        batch_size=batch_size,
        retrain_epochs=args.retrain_epochs,
        lr=args.lr,
        keep_ratio=args.keep_ratio,
        phase5_strategy=args.phase5_strategy,
        phase5_score_norm=args.phase5_score_norm,
        phase5_min_weight=args.phase5_min_weight,
        phase5_weight_power=args.phase5_weight_power,
        random_mode=args.random_mode,
        random_seed=args.random_seed,
        device=device,
        score_field=args.teacher_score_field,
        aux_score_field=args.teacher_aux_score_field,
        dual_noise_end_frac=args.phase5_dual_noise_end_frac,
        dual_ambiguity_start_frac=args.phase5_dual_ambiguity_start_frac,
        dual_ambiguity_peak_frac=args.phase5_dual_ambiguity_peak_frac,
        dual_ambiguity_end_frac=args.phase5_dual_ambiguity_end_frac,
        dual_noise_strength=args.phase5_dual_noise_strength,
        dual_ambiguity_strength=args.phase5_dual_ambiguity_strength,
        attn_implementation=args.esm_attn_implementation,
        esm_torch_dtype=args.esm_torch_dtype,
    )

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

    results = {
        "teacher_run_dir": args.teacher_run_dir,
        "teacher_aux_run_dir": args.teacher_aux_run_dir,
        "resolved_config": resolved,
        "teacher_downstream": downstream_result,
    }
    results_clean = _sanitize(results)
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_clean, f, indent=2)

    logger.info(f"All results saved -> {results_path}")
    logger.info(f"Run directory: {run_dir}")
    logger.info("Teacher-downstream pipeline complete!")


# ==========================================
# Logging Setup
# ==========================================
def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(os.path.join(output_dir, "pipeline.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ==========================================
# Main Pipeline
# ==========================================
def main():
    args = parse_args()
    if args.random_only:
        run_random_only(args)
        return
    if args.teacher_rebuild_only:
        run_teacher_rebuild_only(args)
        return
    if args.teacher_train_only:
        run_teacher_train_only(args)
        return
    if args.teacher_downstream_only:
        run_teacher_downstream_only(args)
        return

    # Parse phases
    phases = set(int(p.strip()) for p in args.phase.split(","))

    # Output dir with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"p4_mixflow_{args.datarater_arch}_{args.outer_sampling}outer"
    run_dir = os.path.join(args.output_dir, f"{run_prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logging(run_dir)
    logger.info("=" * 70)
    logger.info("DataRater Pipeline — Meta-Learned Dataset Curation for PPBA")
    logger.info("=" * 70)
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Phases: {sorted(phases)}")
    logger.info(f"Args: {json.dumps(vars(args), indent=2)}")

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Seed
    _configure_reproducibility(args.seed, strict=args.strict_deterministic)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ==========================================
    # Data Preparation (Shared across phases)
    # ==========================================
    logger.info("\n" + "=" * 50)
    logger.info("Preparing data...")
    logger.info("=" * 50)

    from data_utils import prepare_data

    # Uses the standard base batch_size (64)
    train_loader, val_loader, train_dataset, val_dataset, train_raw_dataset, val_raw_dataset = prepare_data(
        dataset_name=args.dataset,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        deterministic=args.strict_deterministic,
        mode=args.data_mode,
        exclude_sources=_parse_csv_arg(args.exclude_sources),
    )

    results = {}  # Collect all results

    # ==========================================
    # Phase 1: Baseline Training
    # ==========================================
    if 1 in phases:
        logger.info("\n" + "=" * 60)
        logger.info(f"PHASE 1: Baseline Training ({args.epochs} epochs)")
        logger.info("=" * 60)

        from baseline_trainer import train_baseline

        phase1_dir = os.path.join(run_dir, "phase1_baseline")
        sample_auditor = None
        if args.collect_sample_audit:
            from sample_audit import TrainDynamicsAuditor

            sample_auditor = TrainDynamicsAuditor(
                train_dataset=train_dataset,
                raw_train_dataset=train_raw_dataset,
                batch_size=args.batch_size,
                device=device,
                save_dir=os.path.join(phase1_dir, "sample_audit"),
                top_frac=args.audit_top_frac,
                tag="baseline",
                badness_weight_spec=args.teacher_badness_weights,
                noise_weight_spec=args.teacher_noise_weights,
                ambiguity_weight_spec=args.teacher_ambiguity_weights,
            )
        baseline_result = train_baseline(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            save_dir=phase1_dir,
            tag="baseline",
            device=device,
            sample_auditor=sample_auditor,
            attn_implementation=args.esm_attn_implementation,
            esm_torch_dtype=args.esm_torch_dtype,
        )

        results["baseline"] = {
            "best_metrics": baseline_result["best_metrics"],
            "total_flops": baseline_result["total_flops"],
            "history": baseline_result["history"],
        }
        if baseline_result.get("sample_audit") is not None:
            results["sample_audit"] = baseline_result["sample_audit"]

        # Plot training curves
        from viz import plot_training_curves
        plot_training_curves(
            baseline_result["history"],
            title="Phase 1: Baseline Training",
            save_path=os.path.join(run_dir, "plots", "phase1_curves.png"),
        )

        logger.info(f"Phase 1 complete. Best MSE: {baseline_result['best_metrics']['mse']:.4f}")

    # ==========================================
    # Phase 2: Meta-Training DataRater
    # ==========================================
    data_rater = None

    if 2 in phases:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: Meta-Training DataRater")
        logger.info("=" * 60)

        from meta_trainer import run_meta_training
        from data_utils import build_dataloaders
        hard_outer_sources = _parse_csv_arg(args.hard_outer_sources)
        outer_source_weights = _parse_weight_spec(args.outer_source_weights)
        
        # Build specialized dataloaders with smaller meta_batch_size to prevent OOM
        logger.info(f"Building specific DataLoaders for Meta-Training (Batch Size: {args.meta_batch_size})")
        meta_train_loader, meta_val_loader = build_dataloaders(
            train_dataset, val_dataset,
            batch_size=args.meta_batch_size,
            num_workers=args.num_workers,
            seed=args.seed + 1000,
            deterministic=args.strict_deterministic,
        )

        phase2_dir = os.path.join(run_dir, "phase2_datarater")
        meta_result = run_meta_training(
            train_loader=meta_train_loader,
            val_loader=meta_val_loader,
            train_raw=train_raw_dataset,
            val_raw=val_raw_dataset,
            train_dataset=train_dataset,
            n_meta_steps=args.meta_steps,
            n_inner_models=args.n_inner_models,
            lifetime=args.lifetime,
            T_window=args.T_window,
            T_backprop=args.T_backprop,
            temperature=args.temperature,
            outer_objective=args.outer_objective,
            alpha=args.alpha,
            outer_eps=args.outer_eps,
            mse_norm_eps=args.mse_norm_eps,
            use_first_order_ablation=args.ablation,
            sample_one_inner=args.sample_one_inner,
            meta_grad_clip=args.meta_grad_clip,
            canary_interval=args.canary_interval,
            inner_reset_strategy=args.inner_reset_strategy,
            inner_init_bank_dir=args.inner_init_bank_dir,
            inner_init_bank_jitter=args.inner_init_bank_jitter,
            use_zscore_inner=args.use_zscore_inner,
            datarater_arch=args.datarater_arch,
            outer_sampling=args.outer_sampling,
            outer_per_source=args.outer_per_source,
            hard_outer_sources=hard_outer_sources,
            outer_source_weights=outer_source_weights,
            inner_batch_scope=args.inner_batch_scope,
            outer_batch_scope=args.outer_batch_scope,
            num_experts=args.num_experts,
            router_top_k=args.router_top_k,
            capacity_factor=args.capacity_factor,
            router_aux_loss_coef=args.router_aux_loss_coef,
            router_z_loss_coef=args.router_z_loss_coef,
            router_noise_std=args.router_noise_std,
            router_temperature=args.router_temperature,
            moe_score_merge=args.moe_score_merge,
            drop_overflow_tokens=args.drop_overflow_tokens,
            attn_implementation=args.esm_attn_implementation,
            esm_torch_dtype=args.esm_torch_dtype,
            save_dir=phase2_dir,
        )

        data_rater = meta_result["data_rater"]
        results["meta_training"] = {
            "config": meta_result["config"],
            "elapsed": meta_result["elapsed"],
        }

        logger.info(f"Phase 2 complete. Elapsed: {meta_result['elapsed']:.1f}s")

    # Load DataRater from checkpoint if Phase 2 was skipped
    if data_rater is None and args.datarater_ckpt:
        logger.info(f"Loading DataRater from checkpoint: {args.datarater_ckpt}")
        from model import build_datarater_model, infer_source_names
        source_names = infer_source_names(train_raw_dataset, val_raw_dataset)
        data_rater = build_datarater_model(
            arch=args.datarater_arch,
            source_names=source_names,
            attn_implementation=args.esm_attn_implementation,
            esm_torch_dtype=args.esm_torch_dtype,
            num_experts=args.num_experts,
            router_top_k=args.router_top_k,
            capacity_factor=args.capacity_factor,
            router_temperature=args.router_temperature,
            router_noise_std=args.router_noise_std,
            moe_score_merge=args.moe_score_merge,
            drop_overflow_tokens=args.drop_overflow_tokens,
        ).to(device)
        data_rater.load_state_dict(torch.load(args.datarater_ckpt, map_location=device, weights_only=True))

    # ==========================================
    # Phase 3+4: Scoring & Filtering
    # ==========================================
    filtered_dataset = None

    if (3 in phases or 4 in phases) and data_rater is not None:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3+4: Scoring & Filtering")
        logger.info("=" * 60)

        from scoring import run_scoring_and_filtering, score_all_points

        phase34_dir = os.path.join(run_dir, "phase34_scoring")
        filtered_dataset, filter_stats = run_scoring_and_filtering(
            data_rater=data_rater,
            train_dataset=train_dataset,
            raw_train_dataset=train_raw_dataset,
            N_ref=min(args.N_ref, len(train_dataset)),
            B=args.B,
            keep_ratio=args.keep_ratio,
            save_dir=phase34_dir,
        )

        results["filtering"] = filter_stats

        # Plot score distribution
        scores_path = os.path.join(phase34_dir, "all_scores.npy")
        if os.path.exists(scores_path):
            from viz import plot_score_distribution
            scores = np.load(scores_path)
            plot_score_distribution(
                scores,
                save_path=os.path.join(run_dir, "plots", "score_distribution.png"),
            )

        logger.info(f"Phase 3+4 complete. "
                     f"Kept {filter_stats['filtered_size']}/{filter_stats['original_size']} "
                     f"({filter_stats['actual_keep_ratio']:.1%})")
    elif (3 in phases or 4 in phases) and data_rater is None:
        logger.warning("Skipping Phase 3+4: No DataRater available. "
                       "Run Phase 2 or provide --datarater_ckpt.")

    # ==========================================
    # Phase 5: Retrain on Curated Data
    # ==========================================
    if 5 in phases and (filtered_dataset is not None or args.phase5_strategy != "filter"):
        logger.info("\n" + "=" * 60)
        logger.info(
            f"PHASE 5: Retrain with strategy='{args.phase5_strategy}' ({args.retrain_epochs} epochs)"
        )
        logger.info("=" * 60)

        from data_utils import build_dataloaders
        from baseline_trainer import train_baseline

        retrain_train_loader = None
        retrain_loader_factory = None
        phase5_policy_info = None
        phase5_train_size = None

        if args.phase5_strategy == "filter":
            if filtered_dataset is None:
                raise ValueError("Phase 5 strategy 'filter' requires Phase 3+4 filtered_dataset.")
            retrain_train_loader, _ = build_dataloaders(
                filtered_dataset, val_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed + 2000,
                deterministic=args.strict_deterministic,
            )
            phase5_train_size = len(filtered_dataset)
        else:
            if args.phase5_strategy == "dual_curriculum_sampler":
                raise ValueError(
                    "phase5_strategy='dual_curriculum_sampler' is only supported via "
                    "--teacher_downstream_only with --teacher_run_dir and --teacher_aux_run_dir."
                )
            phase34_dir = os.path.join(run_dir, "phase34_scoring")
            scores_path = os.path.join(phase34_dir, "all_scores.npy")
            if not os.path.exists(scores_path):
                raise FileNotFoundError(
                    f"Phase 5 strategy '{args.phase5_strategy}' requires scored training points: {scores_path}"
                )

            from phase5_policy import build_phase5_policy

            all_scores = np.load(scores_path)
            retrain_loader_factory, phase5_policy_info = build_phase5_policy(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                raw_train_dataset=train_raw_dataset,
                scores=all_scores,
                batch_size=args.batch_size,
                strategy=args.phase5_strategy,
                score_norm=args.phase5_score_norm,
                min_weight=args.phase5_min_weight,
                weight_power=args.phase5_weight_power,
            )
            retrain_train_loader = retrain_loader_factory(1, args.retrain_epochs)
            phase5_train_size = len(train_dataset)
            logger.info("Phase 5 score-driven policy: %s", json.dumps(phase5_policy_info, indent=2))

        phase5_dir = os.path.join(run_dir, "phase5_retrained")
        retrained_result = train_baseline(
            train_loader=retrain_train_loader,
            val_loader=val_loader,
            epochs=args.retrain_epochs,
            lr=args.lr,
            save_dir=phase5_dir,
            tag="retrained",
            device=device,
            train_loader_factory=retrain_loader_factory,
            attn_implementation=args.esm_attn_implementation,
            esm_torch_dtype=args.esm_torch_dtype,
        )

        results["retrained"] = {
            "best_metrics": retrained_result["best_metrics"],
            "total_flops": retrained_result["total_flops"],
            "history": retrained_result["history"],
            "phase5_train_size": int(phase5_train_size),
        }
        results["retrained_datarater"] = {
            "phase5_strategy": args.phase5_strategy,
            "metrics": retrained_result["best_metrics"],
        }
        if args.phase5_strategy == "filter":
            results["retrained_datarater"]["keep_ratio"] = float(args.keep_ratio)
            results["retrained_datarater"]["keep_act"] = int(phase5_train_size)
        else:
            results["retrained_datarater"]["phase5_train_size"] = int(phase5_train_size)
        if phase5_policy_info is not None:
            results["retrained_datarater"]["policy"] = phase5_policy_info

        if args.random_baseline and args.phase5_strategy == "filter":
            phase34_dir = os.path.join(run_dir, "phase34_scoring")
            kept_indices_path = os.path.join(phase34_dir, "kept_indices.npy")
            if not os.path.exists(kept_indices_path):
                raise FileNotFoundError(f"Missing kept indices for random baseline: {kept_indices_path}")

            kept_indices = np.load(kept_indices_path)
            keep_act = int(len(kept_indices))
            sources = _extract_sources_for_tokenized(train_dataset, train_raw_dataset)
            random_indices, extra_info = _sample_random_indices(
                mode=args.random_mode,
                keep_act=keep_act,
                keep_ratio=float(args.keep_ratio),
                sources=sources,
                kept_indices=kept_indices,
                seed=args.random_seed,
            )

            random_kept_indices_path = os.path.join(phase34_dir, "random_kept_indices.npy")
            np.save(random_kept_indices_path, random_indices)
            random_info = {
                "mode": args.random_mode,
                "seed": int(args.random_seed),
                "keep_act": keep_act,
                "keep_ratio_target": float(args.keep_ratio),
                **extra_info,
            }
            with open(os.path.join(phase34_dir, "random_info.json"), "w") as f:
                json.dump(random_info, f, indent=2)

            random_subset = train_dataset.select(random_indices.tolist())
            random_loader, _ = build_dataloaders(
                random_subset, val_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.random_seed + 3000,
                deterministic=args.strict_deterministic,
            )
            phase5_random_dir = os.path.join(run_dir, "phase5_random")
            random_result = train_baseline(
                train_loader=random_loader,
                val_loader=val_loader,
                epochs=args.retrain_epochs,
                lr=args.lr,
                save_dir=phase5_random_dir,
                tag="random",
                device=device,
                attn_implementation=args.esm_attn_implementation,
                esm_torch_dtype=args.esm_torch_dtype,
            )
            results["retrained_random"] = {
                "mode": args.random_mode,
                "seed": int(args.random_seed),
                "keep_act": keep_act,
                "metrics": random_result["best_metrics"],
            }
        elif args.random_baseline and args.phase5_strategy != "filter":
            logger.info(
                "Skipping extra random baseline for phase5_strategy='%s'; Phase 1 already provides the uniform full-data control.",
                args.phase5_strategy,
            )

        # Plot retraining curves
        from viz import plot_training_curves
        plot_training_curves(
            retrained_result["history"],
            title=f"Phase 5: Retrained with {args.phase5_strategy}",
            save_path=os.path.join(run_dir, "plots", "phase5_curves.png"),
        )

        # ---- Comparison ----
        if "baseline" in results:
            from viz import plot_comparison, plot_multi_curve_overlay

            plot_comparison(
                baseline_metrics=results["baseline"]["best_metrics"],
                retrained_metrics=results["retrained"]["best_metrics"],
                baseline_flops=results["baseline"]["total_flops"],
                retrained_flops=results["retrained"]["total_flops"],
                save_path=os.path.join(run_dir, "plots", "comparison.png"),
            )

            plot_multi_curve_overlay(
                {
                    "Baseline": results["baseline"]["history"],
                    "Retrained (DataRater)": results["retrained"]["history"],
                },
                metric="val_mse",
                ylabel="Val MSE",
                title="Baseline vs Retrained — Validation MSE",
                save_path=os.path.join(run_dir, "plots", "mse_overlay.png"),
            )

            # Print final comparison
            b = results["baseline"]["best_metrics"]
            r = results["retrained"]["best_metrics"]
            logger.info("\n" + "=" * 60)
            logger.info("FINAL COMPARISON")
            logger.info("=" * 60)
            logger.info(f"{'Metric':<20} {'Baseline':>12} {'Retrained':>12} {'Delta':>12}")
            logger.info("-" * 60)
            for metric in ["mse", "rmse", "pearson_r", "spearman_r"]:
                bv = b.get(metric, 0)
                rv = r.get(metric, 0)
                delta = rv - bv
                logger.info(f"{metric:<20} {bv:>12.4f} {rv:>12.4f} {delta:>+12.4f}")

            bf = results["baseline"]["total_flops"]
            rf = results["retrained"]["total_flops"]
            ppf_b = (1.0 / max(b["mse"], 1e-10)) / max(bf, 1)
            ppf_r = (1.0 / max(r["mse"], 1e-10)) / max(rf, 1)
            logger.info(f"{'total_flops':<20} {bf:>12.2e} {rf:>12.2e} {rf - bf:>+12.2e}")
            logger.info(f"{'perf_per_flop':<20} {ppf_b:>12.2e} {ppf_r:>12.2e} {ppf_r - ppf_b:>+12.2e}")
            logger.info("=" * 60)

    elif 5 in phases and filtered_dataset is None and args.phase5_strategy == "filter":
        logger.warning("Skipping Phase 5: No filtered dataset available.")

    # ==========================================
    # Save Final Results
    # ==========================================
    # Make results JSON-serializable
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

    results_clean = _sanitize(results)
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_clean, f, indent=2)

    logger.info(f"\nAll results saved -> {results_path}")
    logger.info(f"Run directory: {run_dir}")
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
