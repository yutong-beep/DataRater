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
from typing import Dict, Optional, Sequence

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
    datarater_arch: str = "single",
    outer_sampling: str = "random",
    outer_per_source: Optional[int] = None,
    hard_outer_sources: Optional[Sequence[str]] = None,
    num_experts: int = 4,
    router_top_k: int = 2,
    capacity_factor: float = 1.25,
    router_aux_loss_coef: float = 0.01,
    router_z_loss_coef: float = 0.0,
    router_noise_std: float = 0.0,
    router_temperature: float = 1.0,
    moe_score_merge: str = "weighted_sum",
    drop_overflow_tokens: bool = True,
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
        "datarater_arch": datarater_arch,
        "outer_sampling": outer_sampling,
        "outer_per_source": outer_per_source,
        "hard_outer_sources": list(hard_outer_sources) if hard_outer_sources is not None else None,
        "num_experts": num_experts,
        "router_top_k": router_top_k,
        "capacity_factor": capacity_factor,
        "router_aux_loss_coef": router_aux_loss_coef,
        "router_z_loss_coef": router_z_loss_coef,
        "router_noise_std": router_noise_std,
        "router_temperature": router_temperature,
        "moe_score_merge": moe_score_merge,
        "drop_overflow_tokens": drop_overflow_tokens,
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
        datarater_arch=datarater_arch,
        outer_sampling=outer_sampling,
        outer_per_source=outer_per_source,
        hard_outer_sources=hard_outer_sources,
        num_experts=num_experts,
        router_top_k=router_top_k,
        capacity_factor=capacity_factor,
        router_aux_loss_coef=router_aux_loss_coef,
        router_z_loss_coef=router_z_loss_coef,
        router_noise_std=router_noise_std,
        router_temperature=router_temperature,
        moe_score_merge=moe_score_merge,
        drop_overflow_tokens=drop_overflow_tokens,
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
