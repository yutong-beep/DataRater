#!/usr/bin/env python3
import os
import json
import glob
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import EsmModel
from scipy.stats import spearmanr

# Import your existing data utils (no code changes required)
from data_utils import download_and_split, tokenize_dataset

ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
HIDDEN = 320


@dataclass
class Metrics:
    mse: float
    rmse: float
    pearson: float
    spearman: float


class ESMForAffinity(nn.Module):
    """Same style as your baseline: ESM2 + mean pooling + small MLP."""
    def __init__(self, model_name: str = ESM_MODEL_NAME):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # [B,L,H]
        mask = attention_mask.unsqueeze(-1).to(h.dtype)
        summed = (h * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / denom
        pred = self.mlp(pooled).squeeze(-1)
        return pred


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    affinity = torch.tensor([b["affinity"] for b in batch], dtype=torch.float32)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "affinity": affinity}


def find_scores_path(run_dir: str) -> str:
    # Try common location first
    p1 = os.path.join(run_dir, "phase34_scoring", "all_scores.npy")
    if os.path.exists(p1):
        return p1
    # Fallback: search
    hits = glob.glob(os.path.join(run_dir, "**", "all_scores.npy"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"Could not find all_scores.npy under {run_dir}")
    # pick shortest path
    hits.sort(key=len)
    return hits[0]


def stratified_keep_indices_by_source(
    scores: np.ndarray,
    sources: List[str],
    keep_ratio: float
) -> np.ndarray:
    scores = np.asarray(scores)
    sources = np.asarray(sources)

    keep_indices = []
    for s in np.unique(sources):
        idxs = np.where(sources == s)[0]
        k = max(1, int(len(idxs) * keep_ratio))
        local_scores = scores[idxs]
        top_local = np.argsort(local_scores)[-k:]
        keep_indices.extend(list(idxs[top_local]))

    keep_indices = np.array(sorted(keep_indices), dtype=np.int64)
    return keep_indices


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Metrics:
    model.eval()
    preds_all = []
    t_all = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        targets = batch["affinity"].to(device)

        pred = model(input_ids, mask).float()
        preds_all.append(pred.detach().cpu().numpy())
        t_all.append(targets.detach().cpu().numpy())

    y = np.concatenate(preds_all)
    t = np.concatenate(t_all)

    mse = float(np.mean((y - t) ** 2))
    rmse = float(np.sqrt(mse))
    pearson = float(np.corrcoef(y, t)[0, 1]) if (np.std(y) > 1e-12 and np.std(t) > 1e-12) else float("nan")
    spearman = float(spearmanr(y, t).correlation) if (len(y) > 2) else float("nan")
    return Metrics(mse=mse, rmse=rmse, pearson=pearson, spearman=spearman)


def train_retrain_phase5(
    train_ds,
    val_ds,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    num_workers: int = 2,
) -> Tuple[Metrics, Dict[str, Any]]:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn, drop_last=False, pin_memory=True)

    model = ESMForAffinity().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best = None
    best_epoch = -1

    for ep in range(1, epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            targets = batch["affinity"].to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(input_ids, mask).float()
            loss = F.mse_loss(pred, targets, reduction="mean")
            loss.backward()
            opt.step()
            losses.append(loss.item())

        val_metrics = evaluate(model, val_loader, device)
        train_loss = float(np.mean(losses)) if losses else float("nan")

        row = {
            "epoch": ep,
            "train_loss": train_loss,
            "val_mse": val_metrics.mse,
            "val_rmse": val_metrics.rmse,
            "pearson": val_metrics.pearson,
            "spearman": val_metrics.spearman,
        }
        history.append(row)
        print(f"[stratified-phase5] epoch {ep}/{epochs} "
              f"train_loss={train_loss:.4f} val_mse={val_metrics.mse:.4f} "
              f"pearson={val_metrics.pearson:.4f} spearman={val_metrics.spearman:.4f}")

        if best is None or val_metrics.mse < best.mse:
            best = val_metrics
            best_epoch = ep

    return best, {"history": history, "best_epoch": best_epoch}


def try_read_results(run_dir: str) -> Dict[str, Any]:
    p = os.path.join(run_dir, "results.json")
    if not os.path.exists(p):
        return {}
    with open(p, "r") as f:
        return json.load(f)


def extract_metric_block(obj: Dict[str, Any], keys: List[str]) -> Optional[Dict[str, Any]]:
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur if isinstance(cur, dict) else None


def best_effort_get_baseline_and_retrained(results_json: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    # This is best-effort because we don't know your exact schema.
    candidates_baseline = [
        ["baseline"],
        ["phase1"],
        ["phase_1"],
        ["phase1_baseline"],
    ]
    candidates_retrained = [
        ["retrained"],
        ["phase5"],
        ["phase_5"],
        ["phase5_retrained"],
    ]

    baseline = None
    retrained = None
    for path in candidates_baseline:
        block = extract_metric_block(results_json, path)
        if block:
            baseline = block
            break
    for path in candidates_retrained:
        block = extract_metric_block(results_json, path)
        if block:
            retrained = block
            break

    # Sometimes the file is flat with keys like baseline_mse, retrained_mse
    if baseline is None:
        flat = results_json
        if any(k in flat for k in ["baseline_mse", "baseline_rmse", "baseline_pearson_r", "baseline_spearman_r"]):
            baseline = flat
    if retrained is None:
        flat = results_json
        if any(k in flat for k in ["retrained_mse", "retrained_rmse", "retrained_pearson_r", "retrained_spearman_r"]):
            retrained = flat

    return baseline, retrained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="experiments/run_YYYYMMDD_HHMMSS")
    ap.add_argument("--keep_ratio", type=float, default=0.7)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()

    # Repro
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    run_dir = args.run_dir
    scores_path = find_scores_path(run_dir)
    print("loading scores:", scores_path)
    scores = np.load(scores_path)

    # Rebuild the exact same split as the pipeline used
    train_raw, val_raw = download_and_split(
        dataset_name="Bindwell/PPBA",
        train_ratio=args.train_ratio,
        seed=args.seed,
        cache_dir=None,
        mode="combined_train",
    )

    assert len(scores) == len(train_raw), f"scores len {len(scores)} != train len {len(train_raw)}"

    # Tokenize
    train_tok = tokenize_dataset(train_raw, max_length=args.max_length, model_name=ESM_MODEL_NAME)
    val_tok = tokenize_dataset(val_raw, max_length=args.max_length, model_name=ESM_MODEL_NAME)

    # Stratified keep
    sources = list(train_raw["source"])
    keep_idx = stratified_keep_indices_by_source(scores, sources, args.keep_ratio)
    filtered_train = train_tok.select(keep_idx)

    print(f"Stratified keep_ratio={args.keep_ratio} -> kept {len(filtered_train)}/{len(train_tok)}")

    # Save indices
    out_dir = os.path.join(run_dir, "phase5_stratified")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "kept_indices.npy"), keep_idx)

    # Train (phase5-like)
    best_metrics, extra = train_retrain_phase5(
        train_ds=filtered_train,
        val_ds=val_tok,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
    )

    # Load baseline/retrained from original run (best-effort)
    rj = try_read_results(run_dir)
    base_block, retr_block = best_effort_get_baseline_and_retrained(rj)

    def pick(block, key, alt_keys):
        if not isinstance(block, dict):
            return None
        if key in block:
            return block[key]
        for k in alt_keys:
            if k in block:
                return block[k]
        return None

    baseline_mse = pick(base_block, "mse", ["best_val_mse", "val_mse", "baseline_mse"])
    retrained_mse = pick(retr_block, "mse", ["best_val_mse", "val_mse", "retrained_mse"])

    summary = {
        "baseline_mse_from_results_json": baseline_mse,
        "original_phase5_mse_from_results_json": retrained_mse,
        "stratified_phase5_best": {
            "mse": best_metrics.mse,
            "rmse": best_metrics.rmse,
            "pearson": best_metrics.pearson,
            "spearman": best_metrics.spearman,
            "best_epoch": extra["best_epoch"],
        },
        "config": vars(args),
    }

    with open(os.path.join(out_dir, "results_stratified_phase5.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================= COMPARISON (best-effort) =================")
    print(f"Baseline (from {run_dir}/results.json): {baseline_mse}")
    print(f"Original Phase5 (from {run_dir}/results.json): {retrained_mse}")
    print(f"Stratified Phase5 (this script) best MSE: {best_metrics.mse:.4f} (epoch {extra['best_epoch']})")
    print("Saved ->", os.path.join(out_dir, "results_stratified_phase5.json"))
    print("============================================================\n")


if __name__ == "__main__":
    main()