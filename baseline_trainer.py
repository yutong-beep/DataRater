"""
baseline_trainer.py — Phase 1 & Phase 5: Standard supervised training loop.

Train ESMForAffinity on given dataset with MSE loss.
Tracks per-epoch metrics, saves checkpoints, estimates FLOPs.
"""

import os
import time
import json
import logging
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

from model import ESMForAffinity

logger = logging.getLogger(__name__)


# ==========================================
# FLOPs estimation (rough, for comparison)
# ==========================================
def estimate_flops_per_step(model: nn.Module, batch_size: int, seq_len: int) -> float:
    """
    Very rough FLOPs estimate for ESM-8M forward + backward.
    ESM-8M: ~8M params, 6 layers, hidden=320, intermediate=1280.
    Forward ≈ 2 * params * seq_len * batch_size (matrix-multiply dominated).
    Backward ≈ 2x forward.
    """
    n_params = sum(p.numel() for p in model.parameters())
    forward_flops = 2 * n_params * seq_len * batch_size
    backward_flops = 2 * forward_flops  # gradient computation
    return forward_flops + backward_flops


# ==========================================
# Metric computation
# ==========================================
def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute MSE, RMSE, MAE, Pearson, Spearman."""
    mse = float(np.mean((preds - targets) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - targets)))

    try:
        pr, _ = pearsonr(preds, targets)
        sr, _ = spearmanr(preds, targets)
    except Exception:
        pr, sr = 0.0, 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "pearson_r": float(pr),
        "spearman_r": float(sr),
    }


# ==========================================
# Evaluation
# ==========================================
@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Run evaluation, return metrics dict."""
    model.eval()
    all_preds, all_targets = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        targets = batch["affinity"].to(device)

        preds = model(input_ids, mask)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return compute_metrics(preds, targets)


# ==========================================
# Training loop
# ==========================================
def train_baseline(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    max_grad_norm: float = 1.0,
    save_dir: str = "checkpoints/baseline",
    tag: str = "baseline",
    device: Optional[torch.device] = None,
    model: Optional[nn.Module] = None,
    train_loader_factory: Optional[Callable[[int, int], DataLoader]] = None,
    sample_auditor=None,
    attn_implementation: str = "auto",
    esm_torch_dtype: str = "auto",
) -> Dict:
    """
    Standard supervised training.

    Args:
        train_loader: training DataLoader
        val_loader: validation DataLoader
        epochs: number of training epochs
        lr: learning rate
        save_dir: directory to save checkpoints
        tag: name tag for this run
        device: compute device
        model: optional pre-initialized model (otherwise creates new)

    Returns:
        dict with 'model', 'history', 'best_metrics', 'total_flops'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = ESMForAffinity(
            attn_implementation=attn_implementation,
            esm_torch_dtype=esm_torch_dtype,
        ).to(device)
    else:
        model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Estimate FLOPs
    sample_loader = train_loader_factory(1, epochs) if train_loader_factory is not None else train_loader
    sample_batch = next(iter(sample_loader))
    seq_len = sample_batch["input_ids"].shape[1]
    bs = sample_batch["input_ids"].shape[0]
    flops_per_step = estimate_flops_per_step(model, bs, seq_len)
    total_steps = 0

    os.makedirs(save_dir, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[{tag}] Model: {n_params:,} params ({n_trainable:,} trainable)")
    logger.info(f"[{tag}] Training for {epochs} epochs, lr={lr}, bs={bs}, seq_len={seq_len}")
    logger.info(f"[{tag}] Estimated FLOPs/step: {flops_per_step:.2e}")

    history = {
        "train_loss": [],
        "val_mse": [],
        "val_rmse": [],
        "val_pearson": [],
        "val_spearman": [],
        "epoch_time": [],
    }
    best_val_mse = float("inf")
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        current_train_loader = train_loader_factory(epoch, epochs) if train_loader_factory is not None else train_loader
        pbar = tqdm(current_train_loader, desc=f"[{tag}] Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            targets = batch["affinity"].to(device)

            preds = model(input_ids, mask)
            loss = F.mse_loss(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            total_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # ---- Eval ----
        val_metrics = evaluate(model, val_loader, device)

        # Track
        history["train_loss"].append(avg_train_loss)
        history["val_mse"].append(val_metrics["mse"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_pearson"].append(val_metrics["pearson_r"])
        history["val_spearman"].append(val_metrics["spearman_r"])
        history["epoch_time"].append(elapsed)

        if sample_auditor is not None:
            sample_auditor.collect_epoch(model, epoch)

        # Best checkpoint
        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_metrics = val_metrics.copy()
            best_metrics["epoch"] = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, f"{tag}_best.pt"))

        logger.info(
            f"[{tag}] Epoch {epoch}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_mse={val_metrics['mse']:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f} | "
            f"pearson={val_metrics['pearson_r']:.4f} | "
            f"spearman={val_metrics['spearman_r']:.4f} | "
            f"time={elapsed:.1f}s"
        )

    # Save final
    torch.save(model.state_dict(), os.path.join(save_dir, f"{tag}_final.pt"))

    total_flops = total_steps * flops_per_step
    logger.info(f"[{tag}] Training complete. Total FLOPs: {total_flops:.3e}")
    logger.info(f"[{tag}] Best val MSE: {best_val_mse:.4f} (epoch {best_metrics.get('epoch', '?')})")

    # Save history
    with open(os.path.join(save_dir, f"{tag}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    sample_audit_summary = None
    if sample_auditor is not None:
        sample_audit_summary = sample_auditor.finalize()

    return {
        "model": model,
        "history": history,
        "best_metrics": best_metrics,
        "total_flops": total_flops,
        "total_steps": total_steps,
        "flops_per_step": flops_per_step,
        "sample_audit": sample_audit_summary,
    }
