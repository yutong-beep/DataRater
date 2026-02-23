#!/usr/bin/env python3
import os
import json
import glob
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
    def __init__(self, model_name: str = ESM_MODEL_NAME):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask):
        out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.mlp(pooled).squeeze(-1)


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    affinity = torch.tensor([b["affinity"] for b in batch], dtype=torch.float32)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "affinity": affinity}


def find_scores_path(run_dir: str) -> str:
    p = os.path.join(run_dir, "phase34_scoring", "all_scores.npy")
    if os.path.exists(p):
        return p
    hits = glob.glob(os.path.join(run_dir, "**", "all_scores.npy"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"Could not find all_scores.npy under {run_dir}")
    hits.sort(key=len)
    return hits[0]


def stratified_keep_indices_by_source(scores: np.ndarray, sources: List[str], keep_ratio: float) -> np.ndarray:
    scores = np.asarray(scores)
    sources = np.asarray(sources)
    keep = []
    for s in np.unique(sources):
        idxs = np.where(sources == s)[0]
        k = max(1, int(len(idxs) * keep_ratio))
        top = np.argsort(scores[idxs])[-k:]
        keep.extend(list(idxs[top]))
    return np.array(sorted(keep), dtype=np.int64)


def random_stratified_keep_indices_by_source(n: int, sources: List[str], keep_ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sources = np.asarray(sources)
    keep = []
    for s in np.unique(sources):
        idxs = np.where(sources == s)[0]
        k = max(1, int(len(idxs) * keep_ratio))
        chosen = rng.choice(idxs, size=k, replace=False)
        keep.extend(list(chosen))
    return np.array(sorted(keep), dtype=np.int64)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Metrics:
    model.eval()
    ys, ts = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        targets = batch["affinity"].to(device)
        pred = model(input_ids, mask).float()
        ys.append(pred.detach().cpu().numpy())
        ts.append(targets.detach().cpu().numpy())
    y = np.concatenate(ys)
    t = np.concatenate(ts)
    mse = float(np.mean((y - t) ** 2))
    rmse = float(np.sqrt(mse))
    pearson = float(np.corrcoef(y, t)[0, 1]) if (np.std(y) > 1e-12 and np.std(t) > 1e-12) else float("nan")
    spearman = float(spearmanr(y, t).correlation) if (len(y) > 2) else float("nan")
    return Metrics(mse=mse, rmse=rmse, pearson=pearson, spearman=spearman)


def train_phase5_like(train_ds, val_ds, epochs: int, lr: float, batch_size: int, device: torch.device) -> Tuple[Metrics, Dict[str, Any]]:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
                              collate_fn=collate_fn, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2,
                            collate_fn=collate_fn, drop_last=False, pin_memory=True)

    model = ESMForAffinity().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best = None
    best_epoch = -1
    history = []

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
            losses.append(float(loss.item()))

        m = evaluate(model, val_loader, device)
        row = {
            "epoch": ep,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_mse": m.mse,
            "val_rmse": m.rmse,
            "pearson": m.pearson,
            "spearman": m.spearman,
        }
        history.append(row)
        print(f"[phase5] ep {ep}/{epochs} train_loss={row['train_loss']:.4f} val_mse={m.mse:.4f} pearson={m.pearson:.4f} spearman={m.spearman:.4f}")

        if best is None or m.mse < best.mse:
            best = m
            best_epoch = ep

    return best, {"best_epoch": best_epoch, "history": history}


def run_one_setting(
    tag: str,
    keep_idx: np.ndarray,
    train_tok,
    val_tok,
    out_dir: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"kept_indices_{tag}.npy"), keep_idx)
    filtered = train_tok.select(keep_idx)
    print(f"\n=== {tag}: kept {len(filtered)}/{len(train_tok)} ===")
    best, extra = train_phase5_like(filtered, val_tok, epochs=epochs, lr=lr, batch_size=batch_size, device=device)

    result = {
        "tag": tag,
        "kept": int(len(filtered)),
        "total_train": int(len(train_tok)),
        "keep_ratio_effective": float(len(filtered) / len(train_tok)),
        "best": {"mse": best.mse, "rmse": best.rmse, "pearson": best.pearson, "spearman": best.spearman, "best_epoch": extra["best_epoch"]},
        "history": extra["history"],
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
    }
    with open(os.path.join(out_dir, f"result_{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--data_mode", type=str, default="combined_train", choices=["combined_train", "all"])
    ap.add_argument("--keep_ratios", type=str, default="0.7,0.8,0.9,0.95,1.0")
    ap.add_argument("--random_control_ratio", type=float, default=0.7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    run_dir = args.run_dir
    scores_path = find_scores_path(run_dir)
    print("loading scores:", scores_path)
    scores = np.load(scores_path)

    # rebuild same split
    train_raw, val_raw = download_and_split(
        dataset_name="Bindwell/PPBA",
        train_ratio=args.train_ratio,
        seed=args.seed,
        cache_dir=None,
        mode=args.data_mode,
    )

    # tokenize once
    train_tok = tokenize_dataset(train_raw, max_length=args.max_length, model_name=ESM_MODEL_NAME)
    val_tok = tokenize_dataset(val_raw, max_length=args.max_length, model_name=ESM_MODEL_NAME)

    # all_scores.npy is generated from the tokenized training dataset (not raw).
    assert len(scores) == len(train_tok), (
        f"Score length mismatch: scores={len(scores)} vs tokenized_train={len(train_tok)}. "
        "Check seed/data_mode/max_length and make sure scores come from the same run."
    )

    raw_sources = list(train_raw["source"]) if "source" in train_raw.column_names else ["UNKNOWN"] * len(train_raw)
    sources = []
    if "raw_index" in train_tok.column_names:
        for v in train_tok["raw_index"]:
            idx = int(v.item()) if torch.is_tensor(v) else int(v)
            if 0 <= idx < len(raw_sources):
                sources.append(str(raw_sources[idx]))
            else:
                sources.append("UNKNOWN")
    else:
        # Fallback (older tokenized dataset schema): assume no rows were filtered.
        sources = [str(x) for x in raw_sources[:len(train_tok)]]

    sweep_dir = os.path.join(run_dir, "phase5_sweep")
    os.makedirs(sweep_dir, exist_ok=True)

    keep_ratios = [float(x.strip()) for x in args.keep_ratios.split(",") if x.strip()]
    results = []

    # Stratified-by-score sweep
    for kr in keep_ratios:
        if abs(kr - 1.0) < 1e-12:
            keep_idx = np.arange(len(train_tok), dtype=np.int64)  # keep all
            tag = "stratified_score_1.0"
        else:
            keep_idx = stratified_keep_indices_by_source(scores, sources, kr)
            tag = f"stratified_score_{kr:.2f}"
        out_dir = os.path.join(sweep_dir, tag)
        res = run_one_setting(tag, keep_idx, train_tok, val_tok, out_dir, args.epochs, args.lr, args.batch_size, device)
        results.append(res)

    # Random stratified control @ 0.7 by default
    rc = float(args.random_control_ratio)
    keep_idx = random_stratified_keep_indices_by_source(len(train_tok), sources, rc, seed=args.seed)
    tag = f"random_stratified_{rc:.2f}"
    out_dir = os.path.join(sweep_dir, tag)
    res = run_one_setting(tag, keep_idx, train_tok, val_tok, out_dir, args.epochs, args.lr, args.batch_size, device)
    results.append(res)

    # Summary table
    results_sorted = sorted(results, key=lambda r: r["best"]["mse"])
    summary = {
        "run_dir": run_dir,
        "config": vars(args),
        "results": results_sorted,
    }
    with open(os.path.join(sweep_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================== SWEEP SUMMARY (sorted by best MSE) ==================")
    for r in results_sorted:
        b = r["best"]
        print(f"{r['tag']:>22} | kept {r['kept']:>5}/{r['total_train']:<5} "
              f"| best_mse {b['mse']:.4f} | pearson {b['pearson']:.4f} | spearman {b['spearman']:.4f} | epoch {b['best_epoch']}")
    print("Saved ->", os.path.join(sweep_dir, "summary.json"))
    print("=======================================================================\n")


if __name__ == "__main__":
    main()
