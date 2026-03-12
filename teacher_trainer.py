"""
teacher_trainer.py — Supervised DataRater training against teacher-signal labels.
"""

import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import build_datarater_model, datarater_forward, infer_source_names

logger = logging.getLogger(__name__)


def _finite_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    ranks[order] = np.arange(scores.shape[0], dtype=np.float64) + 1.0
    rank_sum_pos = float(ranks[pos].sum())
    return (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)


def _binary_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int64)
    labels_i = labels.astype(np.int64)
    acc = float(np.mean(preds == labels_i))
    pos_rate = float(np.mean(labels_i))
    auc = float(_binary_auc(probs, labels_i))
    bce = float(np.mean(np.maximum(logits, 0.0) - logits * labels_i + np.log1p(np.exp(-np.abs(logits)))))
    return {
        "bce": bce,
        "accuracy": acc,
        "auc": auc,
        "positive_rate": pos_rate,
    }


def _collate_teacher_batch(batch):
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "teacher_label": torch.tensor([float(b["teacher_label"]) for b in batch], dtype=torch.float32),
        "teacher_goodness": torch.tensor([float(b["teacher_goodness"]) for b in batch], dtype=torch.float32),
        "teacher_badness_rank": torch.tensor([float(b["teacher_badness_rank"]) for b in batch], dtype=torch.float32),
    }
    if "raw_index" in batch[0]:
        out["raw_index"] = torch.tensor([int(b["raw_index"]) for b in batch], dtype=torch.long)
    return out


def _make_teacher_loader(dataset, batch_size: int, shuffle: bool, drop_last: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        collate_fn=_collate_teacher_batch,
        drop_last=drop_last,
        pin_memory=True,
    )


def _collate_score_batch(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "raw_index": torch.tensor([int(b["raw_index"]) for b in batch], dtype=torch.long),
    }


def _extract_sources(train_dataset, raw_train_dataset) -> List[str]:
    if raw_train_dataset is None or "source" not in getattr(raw_train_dataset, "column_names", []):
        return ["UNKNOWN"] * len(train_dataset)
    if "raw_index" not in getattr(train_dataset, "column_names", []):
        return ["UNKNOWN"] * len(train_dataset)
    out = []
    n_raw = len(raw_train_dataset)
    for raw_idx_value in train_dataset["raw_index"]:
        raw_idx = int(raw_idx_value.item()) if torch.is_tensor(raw_idx_value) else int(raw_idx_value)
        if 0 <= raw_idx < n_raw:
            out.append(str(raw_train_dataset[raw_idx]["source"]))
        else:
            out.append("UNKNOWN")
    return out


def _prepare_supervised_teacher_datasets(
    train_dataset,
    raw_train_dataset,
    audit_parquet_path: str,
    val_frac: float,
    seed: int,
):
    if not os.path.exists(audit_parquet_path):
        raise FileNotFoundError(f"Teacher audit parquet not found: {audit_parquet_path}")
    if not (0.0 < float(val_frac) < 0.5):
        raise ValueError(f"teacher_val_frac must be in (0, 0.5), got {val_frac}")

    audit_df = pd.read_parquet(audit_parquet_path)
    required_cols = {
        "tokenized_index",
        "raw_index",
        "teacher_label_good",
        "teacher_label_bad",
        "teacher_goodness",
        "teacher_badness_rank",
        "source",
    }
    missing = required_cols.difference(audit_df.columns)
    if missing:
        raise ValueError(f"Teacher audit parquet missing columns: {sorted(missing)}")

    n = len(train_dataset)
    label_full = np.full(n, -1, dtype=np.int64)
    goodness_full = np.full(n, np.nan, dtype=np.float32)
    badness_rank_full = np.full(n, np.nan, dtype=np.float32)

    raw_indices = np.asarray(
        [int(v.item()) if torch.is_tensor(v) else int(v) for v in train_dataset["raw_index"]],
        dtype=np.int64,
    )
    source_full = np.asarray(_extract_sources(train_dataset, raw_train_dataset), dtype=object)

    for row in audit_df.itertuples(index=False):
        idx = int(row.tokenized_index)
        if idx < 0 or idx >= n:
            raise ValueError(f"Audit tokenized_index out of range: {idx}")
        if int(raw_indices[idx]) != int(row.raw_index):
            raise ValueError(
                f"Audit/raw_index mismatch at tokenized_index={idx}: current={int(raw_indices[idx])} audit={int(row.raw_index)}"
            )
        label = -1
        if int(row.teacher_label_good) == 1:
            label = 1
        elif int(row.teacher_label_bad) == 1:
            label = 0
        label_full[idx] = label
        goodness_full[idx] = float(row.teacher_goodness)
        badness_rank_full[idx] = float(row.teacher_badness_rank)

    labeled_idx = np.where(label_full >= 0)[0]
    if labeled_idx.size == 0:
        raise ValueError("No supervised teacher labels found in audit parquet.")

    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    group_keys = np.asarray(
        [f"{source_full[i]}::label{int(label_full[i])}" for i in labeled_idx],
        dtype=object,
    )
    for key in sorted(set(group_keys.tolist())):
        group = labeled_idx[group_keys == key]
        group = group.copy()
        rng.shuffle(group)
        if len(group) <= 1:
            train_idx.extend(group.tolist())
            continue
        n_val = int(round(float(val_frac) * len(group)))
        n_val = max(1, min(len(group) - 1, n_val))
        val_idx.extend(group[:n_val].tolist())
        train_idx.extend(group[n_val:].tolist())

    train_idx = np.array(sorted(train_idx), dtype=np.int64)
    val_idx = np.array(sorted(val_idx), dtype=np.int64)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Teacher split produced an empty train or val partition.")

    supervised_dataset = train_dataset.add_column("teacher_label", label_full.tolist())
    supervised_dataset = supervised_dataset.add_column("teacher_goodness", goodness_full.tolist())
    supervised_dataset = supervised_dataset.add_column("teacher_badness_rank", badness_rank_full.tolist())
    supervised_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "affinity", "raw_index", "teacher_label", "teacher_goodness", "teacher_badness_rank"],
    )

    train_sup = supervised_dataset.select(train_idx.tolist())
    val_sup = supervised_dataset.select(val_idx.tolist())

    split_summary = {
        "num_total_train_samples": int(n),
        "num_labeled_samples": int(labeled_idx.size),
        "num_train_supervised": int(len(train_idx)),
        "num_val_supervised": int(len(val_idx)),
        "positive_rate_train": float(np.mean(label_full[train_idx] == 1)),
        "positive_rate_val": float(np.mean(label_full[val_idx] == 1)),
        "source_counts_train": {
            str(src): int(np.sum(source_full[train_idx] == src))
            for src in sorted(set(source_full[train_idx].tolist()))
        },
        "source_counts_val": {
            str(src): int(np.sum(source_full[val_idx] == src))
            for src in sorted(set(source_full[val_idx].tolist()))
        },
    }
    return train_sup, val_sup, split_summary


@torch.no_grad()
def _evaluate_teacher_model(model, dataloader, device: torch.device, raw_train_dataset) -> Dict[str, float]:
    model.eval()
    logits_all, labels_all, goodness_all, sources_all = [], [], [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        raw_index = batch.get("raw_index")
        if raw_index is not None:
            raw_index = raw_index.to(device)
        logits = datarater_forward(
            model,
            input_ids,
            attention_mask,
            raw_indices=raw_index,
            raw_dataset=raw_train_dataset,
        )
        logits_all.append(logits.detach().cpu().numpy())
        labels_all.append(batch["teacher_label"].detach().cpu().numpy())
        goodness_all.append(batch["teacher_goodness"].detach().cpu().numpy())
        if raw_index is not None and raw_train_dataset is not None and "source" in getattr(raw_train_dataset, "column_names", []):
            for raw_idx in batch["raw_index"].detach().cpu().tolist():
                sources_all.append(str(raw_train_dataset[int(raw_idx)]["source"]))
        else:
            sources_all.extend(["UNKNOWN"] * int(batch["teacher_label"].shape[0]))

    logits = np.concatenate(logits_all).astype(np.float32)
    labels = np.concatenate(labels_all).astype(np.float32)
    teacher_goodness = np.concatenate(goodness_all).astype(np.float32)
    probs = 1.0 / (1.0 + np.exp(-logits))

    metrics = _binary_metrics(logits, labels)
    try:
        metrics["spearman_goodness"] = _finite_float(spearmanr(probs, teacher_goodness).statistic)
    except Exception:
        metrics["spearman_goodness"] = 0.0

    source_metrics = {}
    src_arr = np.asarray(sources_all, dtype=object)
    for src in sorted(set(sources_all)):
        mask = src_arr == src
        src_logits = logits[mask]
        src_labels = labels[mask]
        src_goodness = teacher_goodness[mask]
        source_metrics[str(src)] = {
            "count": int(mask.sum()),
            **_binary_metrics(src_logits, src_labels),
            "spearman_goodness": _finite_float(
                spearmanr(1.0 / (1.0 + np.exp(-src_logits)), src_goodness).statistic
            ) if int(mask.sum()) > 1 else 0.0,
        }
    metrics["source_metrics"] = source_metrics
    return metrics


@torch.no_grad()
def _score_all_train_samples(model, train_dataset, raw_train_dataset, batch_size: int, device: torch.device) -> pd.DataFrame:
    model.eval()
    score_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=_collate_score_batch,
        drop_last=False,
        pin_memory=True,
    )

    logits_all, probs_all, raw_indices_all = [], [], []
    sources_all = []
    for batch in score_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        raw_index = batch["raw_index"].to(device)
        logits = datarater_forward(
            model,
            input_ids,
            attention_mask,
            raw_indices=raw_index,
            raw_dataset=raw_train_dataset,
        )
        probs = torch.sigmoid(logits)
        logits_all.append(logits.detach().cpu().numpy())
        probs_all.append(probs.detach().cpu().numpy())
        raw_idx_cpu = batch["raw_index"].detach().cpu().numpy().astype(np.int64)
        raw_indices_all.append(raw_idx_cpu)
        if raw_train_dataset is not None and "source" in getattr(raw_train_dataset, "column_names", []):
            for raw_idx in raw_idx_cpu.tolist():
                sources_all.append(str(raw_train_dataset[int(raw_idx)]["source"]))
        else:
            sources_all.extend(["UNKNOWN"] * len(raw_idx_cpu))

    raw_indices = np.concatenate(raw_indices_all).astype(np.int64)
    logits = np.concatenate(logits_all).astype(np.float32)
    probs = np.concatenate(probs_all).astype(np.float32)
    return pd.DataFrame(
        {
            "tokenized_index": np.arange(len(raw_indices), dtype=np.int64),
            "raw_index": raw_indices,
            "source": sources_all,
            "pred_logit_good": logits,
            "pred_prob_good": probs,
            "pred_prob_bad": (1.0 - probs).astype(np.float32),
        }
    )


def train_teacher_datarater(
    train_dataset,
    raw_train_dataset,
    audit_parquet_path: str,
    save_dir: str,
    teacher_arch: str = "multihead",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    val_frac: float = 0.2,
    seed: int = 42,
    device: torch.device = None,
    num_experts: int = 4,
    router_top_k: int = 2,
    capacity_factor: float = 1.25,
    router_temperature: float = 1.0,
    router_noise_std: float = 0.0,
    moe_score_merge: str = "weighted_sum",
    drop_overflow_tokens: bool = True,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    train_sup, val_sup, split_summary = _prepare_supervised_teacher_datasets(
        train_dataset=train_dataset,
        raw_train_dataset=raw_train_dataset,
        audit_parquet_path=audit_parquet_path,
        val_frac=val_frac,
        seed=seed,
    )
    with open(os.path.join(save_dir, "split_summary.json"), "w") as f:
        json.dump(split_summary, f, indent=2)

    source_names = infer_source_names(raw_train_dataset)
    model = build_datarater_model(
        arch=teacher_arch,
        source_names=source_names,
        num_experts=num_experts,
        router_top_k=router_top_k,
        capacity_factor=capacity_factor,
        router_temperature=router_temperature,
        router_noise_std=router_noise_std,
        moe_score_merge=moe_score_merge,
        drop_overflow_tokens=drop_overflow_tokens,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_loader = _make_teacher_loader(train_sup, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = _make_teacher_loader(val_sup, batch_size=batch_size, shuffle=False, drop_last=False)

    history = {
        "train_loss": [],
        "val_bce": [],
        "val_accuracy": [],
        "val_auc": [],
        "val_spearman_goodness": [],
    }
    best_auc = -float("inf")
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"[teacher] Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["teacher_label"].to(device)
            raw_index = batch.get("raw_index")
            if raw_index is not None:
                raw_index = raw_index.to(device)

            logits = datarater_forward(
                model,
                input_ids,
                attention_mask,
                raw_indices=raw_index,
                raw_dataset=raw_train_dataset,
            )
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / max(1, n_batches)
        val_metrics = _evaluate_teacher_model(model, val_loader, device, raw_train_dataset)
        history["train_loss"].append(avg_train_loss)
        history["val_bce"].append(val_metrics["bce"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_spearman_goodness"].append(val_metrics["spearman_goodness"])

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_metrics = {**val_metrics, "epoch": int(epoch)}
            torch.save(model.state_dict(), os.path.join(save_dir, "teacher_best.pt"))

        logger.info(
            "[teacher] Epoch %d/%d | train_loss=%.4f | val_bce=%.4f | val_acc=%.4f | val_auc=%.4f | val_spearman=%.4f",
            epoch,
            epochs,
            avg_train_loss,
            val_metrics["bce"],
            val_metrics["accuracy"],
            val_metrics["auc"],
            val_metrics["spearman_goodness"],
        )

    torch.save(model.state_dict(), os.path.join(save_dir, "teacher_final.pt"))
    with open(os.path.join(save_dir, "teacher_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    best_state = torch.load(os.path.join(save_dir, "teacher_best.pt"), map_location=device)
    model.load_state_dict(best_state)
    val_best_metrics = _evaluate_teacher_model(model, val_loader, device, raw_train_dataset)
    val_best_metrics["epoch"] = int(best_metrics.get("epoch", epochs))

    train_score_df = _score_all_train_samples(
        model=model,
        train_dataset=train_dataset,
        raw_train_dataset=raw_train_dataset,
        batch_size=batch_size,
        device=device,
    )
    score_path = os.path.join(save_dir, "teacher_scores.parquet")
    train_score_df.to_parquet(score_path, index=False)

    results = {
        "teacher_arch": teacher_arch,
        "split_summary": split_summary,
        "best_metrics": val_best_metrics,
        "history": history,
        "artifacts": {
            "teacher_best_pt": os.path.join(save_dir, "teacher_best.pt"),
            "teacher_final_pt": os.path.join(save_dir, "teacher_final.pt"),
            "teacher_history_json": os.path.join(save_dir, "teacher_history.json"),
            "teacher_scores_parquet": score_path,
            "split_summary_json": os.path.join(save_dir, "split_summary.json"),
        },
    }
    with open(os.path.join(save_dir, "teacher_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info("[teacher] Saved supervised teacher results -> %s", os.path.join(save_dir, "teacher_results.json"))
    return results
