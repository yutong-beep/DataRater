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
    p.add_argument("--device", type=str, default=None,
                   help="Device (default: auto-detect)")
    p.add_argument("--output_dir", type=str, default="experiments",
                   help="Root output directory")
    p.add_argument("--random_only", action="store_true",
                   help="Run only random-baseline retrain for an existing run_dir, then exit")
    p.add_argument("--run_dir", type=str, default=None,
                   help="Existing run directory used by --random_only")

    # Data
    p.add_argument("--dataset", type=str, default="Bindwell/PPBA")
    p.add_argument("--data_mode", type=str, default="combined_train", choices=["combined_train", "all"],
                   help="Dataset loading mode: 'combined_train' (default) or 'all'")
    p.add_argument("--max_length", type=int, default=512,
                   help="Max sequence length for tokenization")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size for Baseline and Retrain (Phase 1 & 5)")
    p.add_argument("--meta_batch_size", type=int, default=16,
                   help="Smaller batch size specifically for DataRater Meta-Training to prevent OOM (Phase 2)")
    p.add_argument("--train_ratio", type=float, default=0.8)

    # Phase 1 & 5: Baseline training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)

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
                   choices=["mse_norm", "pearson", "cosine", "mix", "source_stratified_mse"],
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
    p.add_argument("--datarater_arch", type=str, default="single",
                   choices=["single", "multihead", "moe"],
                   help="DataRater architecture for Phase 2")
    p.add_argument("--outer_sampling", type=str, default="random",
                   choices=["random", "balanced", "harder"],
                   help="Outer-batch sampling mode for Phase 2")
    p.add_argument("--outer_per_source", type=int, default=None,
                   help="Samples per source in each balanced/harder outer batch (default: auto)")
    p.add_argument("--hard_outer_sources", type=str, default="SKEMPI v2.0,PDBbind v2020",
                   help="Comma-separated source names used when --outer_sampling harder")
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
        mode=args.data_mode,
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
        baseline_result = train_baseline(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            save_dir=phase1_dir,
            tag="baseline",
            device=device,
        )

        results["baseline"] = {
            "best_metrics": baseline_result["best_metrics"],
            "total_flops": baseline_result["total_flops"],
            "history": baseline_result["history"],
        }

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
        
        # Build specialized dataloaders with smaller meta_batch_size to prevent OOM
        logger.info(f"Building specific DataLoaders for Meta-Training (Batch Size: {args.meta_batch_size})")
        meta_train_loader, meta_val_loader = build_dataloaders(
            train_dataset, val_dataset,
            batch_size=args.meta_batch_size,
        )

        phase2_dir = os.path.join(run_dir, "phase2_datarater")
        meta_result = run_meta_training(
            train_loader=meta_train_loader,
            val_loader=meta_val_loader,
            train_raw=train_raw_dataset,
            val_raw=val_raw_dataset,
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
            use_zscore_inner=args.use_zscore_inner,
            datarater_arch=args.datarater_arch,
            outer_sampling=args.outer_sampling,
            outer_per_source=args.outer_per_source,
            hard_outer_sources=hard_outer_sources,
            num_experts=args.num_experts,
            router_top_k=args.router_top_k,
            capacity_factor=args.capacity_factor,
            router_aux_loss_coef=args.router_aux_loss_coef,
            router_z_loss_coef=args.router_z_loss_coef,
            router_noise_std=args.router_noise_std,
            router_temperature=args.router_temperature,
            moe_score_merge=args.moe_score_merge,
            drop_overflow_tokens=args.drop_overflow_tokens,
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
    if 5 in phases and filtered_dataset is not None:
        logger.info("\n" + "=" * 60)
        logger.info(f"PHASE 5: Retrain on DataRater-Curated Data ({args.retrain_epochs} epochs)")
        logger.info("=" * 60)

        from data_utils import build_dataloaders
        from baseline_trainer import train_baseline

        # Uses the standard base batch_size (64)
        filtered_loader, _ = build_dataloaders(
            filtered_dataset, val_dataset,
            batch_size=args.batch_size,
        )

        phase5_dir = os.path.join(run_dir, "phase5_retrained")
        retrained_result = train_baseline(
            train_loader=filtered_loader,
            val_loader=val_loader,
            epochs=args.retrain_epochs,
            lr=args.lr,
            save_dir=phase5_dir,
            tag="retrained",
            device=device,
        )

        results["retrained"] = {
            "best_metrics": retrained_result["best_metrics"],
            "total_flops": retrained_result["total_flops"],
            "history": retrained_result["history"],
            "filtered_train_size": len(filtered_dataset),
        }
        results["retrained_datarater"] = {
            "keep_ratio": float(args.keep_ratio),
            "keep_act": int(len(filtered_dataset)),
            "metrics": retrained_result["best_metrics"],
        }

        if args.random_baseline:
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
            )
            results["retrained_random"] = {
                "mode": args.random_mode,
                "seed": int(args.random_seed),
                "keep_act": keep_act,
                "metrics": random_result["best_metrics"],
            }

        # Plot retraining curves
        from viz import plot_training_curves
        plot_training_curves(
            retrained_result["history"],
            title="Phase 5: Retrained on Curated Data",
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

    elif 5 in phases and filtered_dataset is None:
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
