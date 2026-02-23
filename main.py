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
                   help="Truncated inner loop window")
    p.add_argument("--temperature", type=float, default=0.5,
                   help="Softmax temperature tau used by DataRater in meta-training")
    p.add_argument("--outer_objective", type=str, default="mse_norm",
                   choices=["mse_norm", "pearson", "cosine", "mix"],
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

    # Checkpoints (for resuming)
    p.add_argument("--datarater_ckpt", type=str, default=None,
                   help="Path to pre-trained DataRater checkpoint (skip Phase 2)")

    return p.parse_args()


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

    # Parse phases
    phases = set(int(p.strip()) for p in args.phase.split(","))

    # Output dir with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
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
            temperature=args.temperature,
            outer_objective=args.outer_objective,
            alpha=args.alpha,
            outer_eps=args.outer_eps,
            mse_norm_eps=args.mse_norm_eps,
            use_first_order_ablation=args.ablation,
            sample_one_inner=args.sample_one_inner,
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
        from model import ESMForAffinity
        data_rater = ESMForAffinity().to(device)
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
