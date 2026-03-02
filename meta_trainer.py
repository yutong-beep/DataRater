"""
meta_trainer.py — Wrapper around model.train_datarater with logging,
                   checkpointing, and progress tracking.

Calls the user's train_datarater() with all its parameters and adds:
  - Proper logging
  - Checkpoint saving
  - Config recording
"""

import os
import json
import time
import logging
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from model import train_datarater

logger = logging.getLogger(__name__)


def run_meta_training(
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_raw,
    val_raw,
    n_meta_steps: int = 5000,
    n_inner_models: int = 8,
    lifetime: int = 2000,
    T_window: int = 2,
    T_backprop: int = 2,
    temperature: float = 0.5,
    outer_objective: str = "mse_norm",
    alpha: float = 0.5,
    outer_eps: float = 1e-8,
    mse_norm_eps: float = 1e-6,
    use_first_order_ablation: bool = False,
    sample_one_inner: bool = False,
    use_zscore_inner: bool = False,
    meta_grad_clip: float = 1.0,
    canary_interval: int = 200,
    train_dataset=None,
    save_dir: str = "checkpoints/datarater",
) -> Dict:
    """
    Run DataRater meta-training with logging wrapper.

    All model logic is in model.py (untouched).
    This wrapper adds logging, timing, and checkpoint saving.

    Returns:
        dict with 'data_rater' model and 'config'
    """
    os.makedirs(save_dir, exist_ok=True)

    config = {
        "n_meta_steps": n_meta_steps,
        "n_inner_models": n_inner_models,
        "lifetime": lifetime,
        "T_window": T_window,
        "T_backprop": T_backprop,
        "temperature": temperature,
        "outer_objective": outer_objective,
        "alpha": alpha,
        "outer_eps": outer_eps,
        "mse_norm_eps": mse_norm_eps,
        "use_first_order_ablation": use_first_order_ablation,
        "sample_one_inner": sample_one_inner,
        "use_zscore_inner": use_zscore_inner,
        "meta_grad_clip": meta_grad_clip,
        "canary_interval": canary_interval,
    }

    logger.info("=" * 60)
    logger.info("Phase 2: Meta-Training DataRater")
    logger.info("=" * 60)
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Val loader: {len(val_loader)} batches")

    t0 = time.time()

    # Call the core meta-training function (from model.py, UNCHANGED)
    data_rater = train_datarater(
        train_loader=train_loader,
        val_loader=val_loader,
        n_meta_steps=n_meta_steps,
        n_inner_models=n_inner_models,
        lifetime=lifetime,
        T_window=T_window,
        T_backprop=T_backprop,
        tau=temperature,
        outer_objective=outer_objective,
        alpha=alpha,
        outer_eps=outer_eps,
        mse_norm_eps=mse_norm_eps,
        train_raw=train_raw,
        val_raw=val_raw,
        use_first_order_ablation=use_first_order_ablation,
        sample_one_inner=sample_one_inner,
        use_zscore_inner=use_zscore_inner,
        meta_grad_clip=meta_grad_clip,
        canary_interval=canary_interval,
        train_dataset=train_dataset,
    )

    elapsed = time.time() - t0
    logger.info(f"Meta-training complete in {elapsed:.1f}s ({elapsed / 60:.1f}min)")

    # Save DataRater checkpoint
    ckpt_path = os.path.join(save_dir, "datarater.pt")
    torch.save(data_rater.state_dict(), ckpt_path)
    logger.info(f"DataRater checkpoint saved -> {ckpt_path}")

    # Save config
    with open(os.path.join(save_dir, "meta_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return {
        "data_rater": data_rater,
        "config": config,
        "elapsed": elapsed,
    }
