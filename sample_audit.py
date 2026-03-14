"""
sample_audit.py — Training-dynamics audit and teacher-signal export.

Collects per-sample train-set predictions/losses across epochs and writes a
source-aware teacher cache for downstream data-quality experiments.
"""

import json
import logging
import os
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_utils import build_single_loader

logger = logging.getLogger(__name__)


FORMULA_FEATURE_REGISTRY = {
    "rank_mean_sq_error": "rank_mean_sq_error",
    "rank_final_sq_error": "rank_final_sq_error",
    "rank_std_sq_error": "rank_std_sq_error",
    "rank_late_mean_sq_error": "rank_late_mean_sq_error",
    "rank_sq_error_volatility": "rank_sq_error_volatility",
    "rank_prediction_std": "rank_prediction_std",
    "rank_improvement_ratio": "rank_improvement_ratio",
    "one_minus_rank_improvement_ratio": "__derived__",
    "mid_difficulty": "mid_difficulty",
}

DEFAULT_TEACHER_FORMULAS = {
    "badness": {
        "rank_mean_sq_error": 0.5,
        "rank_final_sq_error": 0.3,
        "rank_std_sq_error": 0.2,
    },
    "noise_risk": {
        "rank_late_mean_sq_error": 0.40,
        "rank_final_sq_error": 0.20,
        "rank_sq_error_volatility": 0.15,
        "rank_prediction_std": 0.15,
        "one_minus_rank_improvement_ratio": 0.10,
    },
    "ambiguity": {
        "rank_prediction_std": 0.35,
        "rank_sq_error_volatility": 0.25,
        "mid_difficulty": 0.20,
        "rank_improvement_ratio": 0.20,
    },
}


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float32)
    ranks[order] = np.arange(values.size, dtype=np.float32)
    if values.size == 1:
        return np.ones(1, dtype=np.float32)
    return ranks / float(values.size - 1)


def _within_source_ranks(values: np.ndarray, sources: List[str]) -> np.ndarray:
    src_arr = np.asarray(sources, dtype=object)
    out = np.zeros(values.shape[0], dtype=np.float32)
    for src in sorted(set(sources)):
        mask = src_arr == src
        out[mask] = _percentile_ranks(values[mask])
    return out


def _epoch_slope(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] <= 1:
        return np.zeros(matrix.shape[1], dtype=np.float32)
    t = np.arange(matrix.shape[0], dtype=np.float32)
    t_centered = t - float(np.mean(t))
    denom = float(np.sum(t_centered ** 2))
    if denom <= 0.0:
        return np.zeros(matrix.shape[1], dtype=np.float32)
    y_centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    slope = np.sum(t_centered[:, None] * y_centered, axis=0) / denom
    return slope.astype(np.float32)


def _extract_sources(train_dataset, raw_train_dataset) -> List[str]:
    if raw_train_dataset is None or "source" not in getattr(raw_train_dataset, "column_names", []):
        return ["UNKNOWN"] * len(train_dataset)
    if "raw_index" not in getattr(train_dataset, "column_names", []):
        return ["UNKNOWN"] * len(train_dataset)

    sources = []
    n_raw = len(raw_train_dataset)
    for raw_idx_value in train_dataset["raw_index"]:
        raw_idx = int(raw_idx_value.item()) if torch.is_tensor(raw_idx_value) else int(raw_idx_value)
        if 0 <= raw_idx < n_raw:
            sources.append(str(raw_train_dataset[raw_idx]["source"]))
        else:
            sources.append("UNKNOWN")
    return sources


def _normalize_formula_weights(weight_map: Dict[str, float], signal_name: str) -> Dict[str, float]:
    if not weight_map:
        raise ValueError(f"{signal_name} formula must contain at least one feature.")
    out = {}
    total = 0.0
    for feature_name, raw_weight in weight_map.items():
        if feature_name not in FORMULA_FEATURE_REGISTRY:
            raise ValueError(
                f"Unsupported {signal_name} formula feature '{feature_name}'. "
                f"Valid options: {sorted(FORMULA_FEATURE_REGISTRY)}"
            )
        weight = float(raw_weight)
        if weight < 0.0:
            raise ValueError(f"{signal_name} formula weight for '{feature_name}' must be non-negative, got {weight}.")
        if weight == 0.0:
            continue
        out[feature_name] = weight
        total += weight
    if total <= 0.0:
        raise ValueError(f"{signal_name} formula weights must sum to a positive value.")
    return {feature_name: (weight / total) for feature_name, weight in out.items()}


def parse_teacher_formula_spec(weight_spec: Optional[str], signal_name: str) -> Dict[str, float]:
    defaults = DEFAULT_TEACHER_FORMULAS[signal_name]
    if weight_spec is None or str(weight_spec).strip() == "":
        return defaults.copy()

    parsed = {}
    for raw_part in str(weight_spec).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Invalid {signal_name} formula part '{part}'. "
                "Use comma-separated feature=weight entries."
            )
        feature_name, raw_weight = part.split("=", 1)
        feature_name = feature_name.strip()
        raw_weight = raw_weight.strip()
        if not feature_name or not raw_weight:
            raise ValueError(
                f"Invalid {signal_name} formula part '{part}'. "
                "Each entry must look like feature=weight."
            )
        parsed[feature_name] = float(raw_weight)
    return _normalize_formula_weights(parsed, signal_name=signal_name)


def _resolve_formula_config(
    badness_weight_spec: Optional[str],
    noise_weight_spec: Optional[str],
    ambiguity_weight_spec: Optional[str],
) -> Dict[str, Dict[str, float]]:
    return {
        "badness": parse_teacher_formula_spec(badness_weight_spec, signal_name="badness"),
        "noise_risk": parse_teacher_formula_spec(noise_weight_spec, signal_name="noise_risk"),
        "ambiguity": parse_teacher_formula_spec(ambiguity_weight_spec, signal_name="ambiguity"),
    }


def _compute_formula_features(audit_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    required = {
        "rank_mean_sq_error",
        "rank_final_sq_error",
        "rank_std_sq_error",
        "rank_late_mean_sq_error",
        "rank_sq_error_volatility",
        "rank_prediction_std",
        "rank_improvement_ratio",
    }
    missing = required.difference(audit_df.columns)
    if missing:
        raise ValueError(f"Audit parquet missing formula feature columns: {sorted(missing)}")

    rank_mean_sq = audit_df["rank_mean_sq_error"].to_numpy(dtype=np.float32)
    rank_final_sq = audit_df["rank_final_sq_error"].to_numpy(dtype=np.float32)
    rank_std_sq = audit_df["rank_std_sq_error"].to_numpy(dtype=np.float32)
    rank_late_mean_sq = audit_df["rank_late_mean_sq_error"].to_numpy(dtype=np.float32)
    rank_sq_vol = audit_df["rank_sq_error_volatility"].to_numpy(dtype=np.float32)
    rank_pred_std = audit_df["rank_prediction_std"].to_numpy(dtype=np.float32)
    rank_improvement = audit_df["rank_improvement_ratio"].to_numpy(dtype=np.float32)
    if "mid_difficulty" in audit_df.columns:
        mid_difficulty = audit_df["mid_difficulty"].to_numpy(dtype=np.float32)
    else:
        mid_difficulty = (1.0 - np.abs((2.0 * rank_late_mean_sq) - 1.0)).astype(np.float32)

    return {
        "rank_mean_sq_error": rank_mean_sq,
        "rank_final_sq_error": rank_final_sq,
        "rank_std_sq_error": rank_std_sq,
        "rank_late_mean_sq_error": rank_late_mean_sq,
        "rank_sq_error_volatility": rank_sq_vol,
        "rank_prediction_std": rank_pred_std,
        "rank_improvement_ratio": rank_improvement,
        "one_minus_rank_improvement_ratio": (1.0 - rank_improvement).astype(np.float32),
        "mid_difficulty": mid_difficulty.astype(np.float32),
    }


def apply_teacher_formulas(
    audit_df: pd.DataFrame,
    top_frac: float,
    badness_weight_spec: Optional[str] = None,
    noise_weight_spec: Optional[str] = None,
    ambiguity_weight_spec: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if not (0.0 < float(top_frac) < 0.5):
        raise ValueError(f"top_frac must be in (0, 0.5), got {top_frac}")
    if "source" not in audit_df.columns:
        raise ValueError("Audit parquet must contain a 'source' column.")

    formula_config = _resolve_formula_config(
        badness_weight_spec=badness_weight_spec,
        noise_weight_spec=noise_weight_spec,
        ambiguity_weight_spec=ambiguity_weight_spec,
    )
    features = _compute_formula_features(audit_df)
    sources = [str(v) for v in audit_df["source"].tolist()]

    def _combine(signal_name: str) -> np.ndarray:
        weights = formula_config[signal_name]
        score = np.zeros(len(audit_df), dtype=np.float32)
        for feature_name, weight in weights.items():
            score += float(weight) * features[feature_name]
        return score.astype(np.float32)

    teacher_badness = _combine("badness")
    teacher_badness_rank = _within_source_ranks(teacher_badness, sources)
    teacher_goodness = (1.0 - teacher_badness_rank).astype(np.float32)

    teacher_noise_risk = _combine("noise_risk")
    teacher_noise_risk_rank = _within_source_ranks(teacher_noise_risk, sources)

    teacher_ambiguity = _combine("ambiguity")
    teacher_ambiguity_rank = _within_source_ranks(teacher_ambiguity, sources)

    is_bad = teacher_badness_rank >= (1.0 - float(top_frac))
    is_good = teacher_badness_rank <= float(top_frac)
    is_noisy = teacher_noise_risk_rank >= (1.0 - float(top_frac))
    is_ambiguous = teacher_ambiguity_rank >= (1.0 - float(top_frac))

    out_df = audit_df.copy()
    out_df["teacher_badness"] = teacher_badness.astype(np.float32)
    out_df["teacher_badness_rank"] = teacher_badness_rank.astype(np.float32)
    out_df["teacher_goodness"] = teacher_goodness.astype(np.float32)
    out_df["teacher_noise_risk"] = teacher_noise_risk.astype(np.float32)
    out_df["teacher_noise_risk_rank"] = teacher_noise_risk_rank.astype(np.float32)
    out_df["teacher_ambiguity"] = teacher_ambiguity.astype(np.float32)
    out_df["teacher_ambiguity_rank"] = teacher_ambiguity_rank.astype(np.float32)
    out_df["teacher_label_bad"] = is_bad.astype(np.int8)
    out_df["teacher_label_good"] = is_good.astype(np.int8)
    out_df["teacher_label_noisy"] = is_noisy.astype(np.int8)
    out_df["teacher_label_ambiguous"] = is_ambiguous.astype(np.int8)

    source_stats = {}
    for src, src_df in out_df.groupby("source", sort=True):
        source_stats[str(src)] = {
            "count": int(len(src_df)),
            "mean_sq_error": float(src_df["mean_sq_error"].mean()),
            "final_sq_error": float(src_df["final_sq_error"].mean()),
            "teacher_bad_rate": float(src_df["teacher_label_bad"].mean()),
            "teacher_good_rate": float(src_df["teacher_label_good"].mean()),
            "teacher_noisy_rate": float(src_df["teacher_label_noisy"].mean()),
            "teacher_ambiguous_rate": float(src_df["teacher_label_ambiguous"].mean()),
        }

    summary = {
        "top_frac": float(top_frac),
        "formula_features": sorted(features.keys()),
        "formula_config": formula_config,
        "global_stats": {
            "mean_sq_error": float(out_df["mean_sq_error"].mean()),
            "final_sq_error": float(out_df["final_sq_error"].mean()),
            "teacher_bad_rate": float(out_df["teacher_label_bad"].mean()),
            "teacher_good_rate": float(out_df["teacher_label_good"].mean()),
            "teacher_noisy_rate": float(out_df["teacher_label_noisy"].mean()),
            "teacher_ambiguous_rate": float(out_df["teacher_label_ambiguous"].mean()),
        },
        "source_stats": source_stats,
    }
    return out_df, summary


def rebuild_teacher_cache(
    source_audit_run_dir: str,
    rebuilt_run_dir: str,
    top_frac: Optional[float] = None,
    badness_weight_spec: Optional[str] = None,
    noise_weight_spec: Optional[str] = None,
    ambiguity_weight_spec: Optional[str] = None,
) -> Dict[str, object]:
    source_sample_dir = os.path.join(source_audit_run_dir, "phase1_baseline", "sample_audit")
    source_parquet = os.path.join(source_sample_dir, "sample_audit.parquet")
    if not os.path.exists(source_parquet):
        raise FileNotFoundError(f"Source audit parquet not found: {source_parquet}")

    source_summary_path = os.path.join(source_sample_dir, "teacher_signal_summary.json")
    source_summary = json.load(open(source_summary_path)) if os.path.exists(source_summary_path) else {}
    rebuild_top_frac = float(source_summary.get("top_frac", 0.2) if top_frac is None else top_frac)

    audit_df = pd.read_parquet(source_parquet)
    rebuilt_df, rebuilt_summary = apply_teacher_formulas(
        audit_df,
        top_frac=rebuild_top_frac,
        badness_weight_spec=badness_weight_spec,
        noise_weight_spec=noise_weight_spec,
        ambiguity_weight_spec=ambiguity_weight_spec,
    )

    rebuilt_sample_dir = os.path.join(rebuilt_run_dir, "phase1_baseline", "sample_audit")
    os.makedirs(rebuilt_sample_dir, exist_ok=True)

    rebuilt_parquet = os.path.join(rebuilt_sample_dir, "sample_audit.parquet")
    rebuilt_df.to_parquet(rebuilt_parquet, index=False)

    for filename in [
        "train_eval_predictions_by_epoch.npy",
        "train_eval_sq_error_by_epoch.npy",
        "train_eval_abs_error_by_epoch.npy",
        "audit_epoch_metrics.json",
    ]:
        src = os.path.join(source_sample_dir, filename)
        dst = os.path.join(rebuilt_sample_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    rebuilt_summary.update(
        {
            "num_samples": int(len(rebuilt_df)),
            "audit_epochs": int(source_summary.get("audit_epochs", 0)),
            "source_audit_run_dir": source_audit_run_dir,
            "artifacts": {
                "sample_audit_parquet": rebuilt_parquet,
                "predictions_by_epoch_npy": os.path.join(rebuilt_sample_dir, "train_eval_predictions_by_epoch.npy"),
                "sq_error_by_epoch_npy": os.path.join(rebuilt_sample_dir, "train_eval_sq_error_by_epoch.npy"),
                "abs_error_by_epoch_npy": os.path.join(rebuilt_sample_dir, "train_eval_abs_error_by_epoch.npy"),
                "audit_epoch_metrics_json": os.path.join(rebuilt_sample_dir, "audit_epoch_metrics.json"),
            },
        }
    )
    rebuilt_summary_path = os.path.join(rebuilt_sample_dir, "teacher_signal_summary.json")
    with open(rebuilt_summary_path, "w") as f:
        json.dump(rebuilt_summary, f, indent=2)

    source_cfg_path = os.path.join(source_audit_run_dir, "config.json")
    cfg_payload = json.load(open(source_cfg_path)) if os.path.exists(source_cfg_path) else {}
    cfg_payload = dict(cfg_payload)
    cfg_payload["teacher_rebuild"] = {
        "source_audit_run_dir": source_audit_run_dir,
        "top_frac": rebuild_top_frac,
        "formula_config": rebuilt_summary["formula_config"],
    }
    with open(os.path.join(rebuilt_run_dir, "config.json"), "w") as f:
        json.dump(cfg_payload, f, indent=2)

    source_results_path = os.path.join(source_audit_run_dir, "results.json")
    source_results = json.load(open(source_results_path)) if os.path.exists(source_results_path) else {}
    rebuilt_results = {
        "source_audit_run_dir": source_audit_run_dir,
        "baseline": source_results.get("baseline", {}),
        "resolved_config": source_results.get("resolved_config", {}),
        "sample_audit": rebuilt_summary,
    }
    with open(os.path.join(rebuilt_run_dir, "results.json"), "w") as f:
        json.dump(rebuilt_results, f, indent=2)

    logger.info("Rebuilt teacher cache -> %s", rebuilt_parquet)
    return {
        "rebuilt_run_dir": rebuilt_run_dir,
        "sample_audit_parquet": rebuilt_parquet,
        "summary": rebuilt_summary,
    }


class TrainDynamicsAuditor:
    def __init__(
        self,
        train_dataset,
        raw_train_dataset,
        batch_size: int,
        device: torch.device,
        save_dir: str,
        top_frac: float = 0.2,
        num_workers: int = 2,
        tag: str = "baseline",
        badness_weight_spec: Optional[str] = None,
        noise_weight_spec: Optional[str] = None,
        ambiguity_weight_spec: Optional[str] = None,
    ):
        if "raw_index" not in getattr(train_dataset, "column_names", []):
            raise ValueError("TrainDynamicsAuditor requires tokenized train_dataset with raw_index.")

        self.train_dataset = train_dataset
        self.raw_train_dataset = raw_train_dataset
        self.batch_size = int(batch_size)
        self.device = device
        self.save_dir = save_dir
        self.top_frac = float(top_frac)
        if not (0.0 < self.top_frac < 0.5):
            raise ValueError(f"top_frac must be in (0, 0.5), got {self.top_frac}")
        self.tag = tag
        self.badness_weight_spec = badness_weight_spec
        self.noise_weight_spec = noise_weight_spec
        self.ambiguity_weight_spec = ambiguity_weight_spec
        self.audit_loader = build_single_loader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
        )

        self.raw_indices = np.asarray(
            [int(v.item()) if torch.is_tensor(v) else int(v) for v in train_dataset["raw_index"]],
            dtype=np.int64,
        )
        self.targets = np.asarray(
            [float(v.item()) if torch.is_tensor(v) else float(v) for v in train_dataset["affinity"]],
            dtype=np.float32,
        )
        self.sources = _extract_sources(train_dataset, raw_train_dataset)
        self.raw_to_pos = {int(raw_idx): pos for pos, raw_idx in enumerate(self.raw_indices.tolist())}
        if len(self.raw_to_pos) != len(self.raw_indices):
            raise ValueError("TrainDynamicsAuditor requires unique raw_index per tokenized sample.")

        self.epoch_predictions: List[np.ndarray] = []
        self.epoch_squared_errors: List[np.ndarray] = []
        self.epoch_abs_errors: List[np.ndarray] = []
        self.epoch_metrics: List[Dict] = []

    @torch.no_grad()
    def collect_epoch(self, model: torch.nn.Module, epoch: int) -> Dict:
        model.eval()

        n = len(self.train_dataset)
        preds_out = np.zeros(n, dtype=np.float32)
        sq_out = np.zeros(n, dtype=np.float32)
        abs_out = np.zeros(n, dtype=np.float32)

        seen = np.zeros(n, dtype=np.int8)
        pbar = tqdm(self.audit_loader, desc=f"[audit:{self.tag}] train-eval {epoch}", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["affinity"].to(self.device)
            raw_index = batch.get("raw_index")
            if raw_index is None:
                raise ValueError("TrainDynamicsAuditor expected raw_index in audit batches.")

            batch_preds = model(input_ids, attention_mask)
            batch_sq = (batch_preds - targets) ** 2
            batch_abs = torch.abs(batch_preds - targets)

            pred_np = batch_preds.detach().cpu().numpy().astype(np.float32)
            sq_np = batch_sq.detach().cpu().numpy().astype(np.float32)
            abs_np = batch_abs.detach().cpu().numpy().astype(np.float32)
            raw_np = raw_index.detach().cpu().numpy().astype(np.int64)

            for raw_idx, pred_val, sq_val, abs_val in zip(raw_np.tolist(), pred_np.tolist(), sq_np.tolist(), abs_np.tolist()):
                pos = self.raw_to_pos[int(raw_idx)]
                preds_out[pos] = float(pred_val)
                sq_out[pos] = float(sq_val)
                abs_out[pos] = float(abs_val)
                seen[pos] = 1

        if int(seen.sum()) != n:
            raise ValueError(f"Audit pass covered {int(seen.sum())}/{n} samples.")

        metrics = {
            "epoch": int(epoch),
            "train_eval_mse": float(np.mean(sq_out)),
            "train_eval_rmse": float(np.sqrt(np.mean(sq_out))),
            "train_eval_mae": float(np.mean(abs_out)),
        }
        self.epoch_predictions.append(preds_out)
        self.epoch_squared_errors.append(sq_out)
        self.epoch_abs_errors.append(abs_out)
        self.epoch_metrics.append(metrics)
        logger.info(
            "[audit:%s] Epoch %d | train_eval_mse=%.4f | train_eval_rmse=%.4f | train_eval_mae=%.4f",
            self.tag,
            epoch,
            metrics["train_eval_mse"],
            metrics["train_eval_rmse"],
            metrics["train_eval_mae"],
        )
        return metrics

    def finalize(self) -> Dict:
        if not self.epoch_squared_errors:
            raise ValueError("No audit epochs were collected.")

        os.makedirs(self.save_dir, exist_ok=True)

        pred_matrix = np.stack(self.epoch_predictions, axis=0).astype(np.float32)
        sq_matrix = np.stack(self.epoch_squared_errors, axis=0).astype(np.float32)
        abs_matrix = np.stack(self.epoch_abs_errors, axis=0).astype(np.float32)

        mean_sq = sq_matrix.mean(axis=0)
        std_sq = sq_matrix.std(axis=0)
        min_sq = sq_matrix.min(axis=0)
        max_sq = sq_matrix.max(axis=0)
        first_sq = sq_matrix[0]
        final_sq = sq_matrix[-1]
        mean_abs = abs_matrix.mean(axis=0)
        final_pred = pred_matrix[-1]
        mean_pred = pred_matrix.mean(axis=0)
        loss_improvement = first_sq - final_sq
        audit_epochs = int(sq_matrix.shape[0])
        early_len = max(1, audit_epochs // 3)
        late_len = max(1, audit_epochs // 3)
        early_mean_sq = sq_matrix[:early_len].mean(axis=0)
        late_mean_sq = sq_matrix[-late_len:].mean(axis=0)
        sq_error_slope = _epoch_slope(sq_matrix)
        pred_slope = _epoch_slope(pred_matrix)
        if audit_epochs > 1:
            sq_error_volatility = np.mean(np.abs(np.diff(sq_matrix, axis=0)), axis=0).astype(np.float32)
            pred_volatility = np.mean(np.abs(np.diff(pred_matrix, axis=0)), axis=0).astype(np.float32)
        else:
            sq_error_volatility = np.zeros(len(self.train_dataset), dtype=np.float32)
            pred_volatility = np.zeros(len(self.train_dataset), dtype=np.float32)
        pred_std = pred_matrix.std(axis=0).astype(np.float32)
        pred_drift = np.abs(pred_matrix[-1] - pred_matrix[0]).astype(np.float32)
        improvement_ratio = ((early_mean_sq - late_mean_sq) / np.maximum(early_mean_sq, 1e-6)).astype(np.float32)

        rank_mean_sq = _within_source_ranks(mean_sq, self.sources)
        rank_final_sq = _within_source_ranks(final_sq, self.sources)
        rank_std_sq = _within_source_ranks(std_sq, self.sources)
        rank_late_mean_sq = _within_source_ranks(late_mean_sq, self.sources)
        rank_sq_error_volatility = _within_source_ranks(sq_error_volatility, self.sources)
        rank_pred_std = _within_source_ranks(pred_std, self.sources)
        rank_improvement_ratio = _within_source_ranks(improvement_ratio, self.sources)
        mid_difficulty = (1.0 - np.abs((2.0 * rank_late_mean_sq) - 1.0)).astype(np.float32)

        audit_df = pd.DataFrame(
            {
                "tokenized_index": np.arange(len(self.train_dataset), dtype=np.int64),
                "raw_index": self.raw_indices,
                "source": self.sources,
                "target_affinity": self.targets,
                "mean_sq_error": mean_sq.astype(np.float32),
                "std_sq_error": std_sq.astype(np.float32),
                "min_sq_error": min_sq.astype(np.float32),
                "max_sq_error": max_sq.astype(np.float32),
                "first_sq_error": first_sq.astype(np.float32),
                "final_sq_error": final_sq.astype(np.float32),
                "early_mean_sq_error": early_mean_sq.astype(np.float32),
                "late_mean_sq_error": late_mean_sq.astype(np.float32),
                "mean_abs_error": mean_abs.astype(np.float32),
                "sq_error_volatility": sq_error_volatility.astype(np.float32),
                "sq_error_slope": sq_error_slope.astype(np.float32),
                "final_prediction": final_pred.astype(np.float32),
                "mean_prediction": mean_pred.astype(np.float32),
                "prediction_std": pred_std.astype(np.float32),
                "prediction_drift": pred_drift.astype(np.float32),
                "prediction_volatility": pred_volatility.astype(np.float32),
                "prediction_slope": pred_slope.astype(np.float32),
                "loss_improvement": loss_improvement.astype(np.float32),
                "improvement_ratio": improvement_ratio.astype(np.float32),
                "rank_mean_sq_error": rank_mean_sq.astype(np.float32),
                "rank_final_sq_error": rank_final_sq.astype(np.float32),
                "rank_std_sq_error": rank_std_sq.astype(np.float32),
                "rank_late_mean_sq_error": rank_late_mean_sq.astype(np.float32),
                "rank_sq_error_volatility": rank_sq_error_volatility.astype(np.float32),
                "rank_prediction_std": rank_pred_std.astype(np.float32),
                "rank_improvement_ratio": rank_improvement_ratio.astype(np.float32),
                "mid_difficulty": mid_difficulty.astype(np.float32),
            }
        )
        audit_df, teacher_summary = apply_teacher_formulas(
            audit_df,
            top_frac=self.top_frac,
            badness_weight_spec=self.badness_weight_spec,
            noise_weight_spec=self.noise_weight_spec,
            ambiguity_weight_spec=self.ambiguity_weight_spec,
        )

        audit_path = os.path.join(self.save_dir, "sample_audit.parquet")
        pred_path = os.path.join(self.save_dir, "train_eval_predictions_by_epoch.npy")
        sq_path = os.path.join(self.save_dir, "train_eval_sq_error_by_epoch.npy")
        abs_path = os.path.join(self.save_dir, "train_eval_abs_error_by_epoch.npy")
        metrics_path = os.path.join(self.save_dir, "audit_epoch_metrics.json")
        summary_path = os.path.join(self.save_dir, "teacher_signal_summary.json")

        audit_df.to_parquet(audit_path, index=False)
        np.save(pred_path, pred_matrix)
        np.save(sq_path, sq_matrix)
        np.save(abs_path, abs_matrix)
        with open(metrics_path, "w") as f:
            json.dump(self.epoch_metrics, f, indent=2)

        summary = {
            "num_samples": int(len(audit_df)),
            "audit_epochs": int(len(self.epoch_metrics)),
            "top_frac": float(self.top_frac),
            "artifacts": {
                "sample_audit_parquet": audit_path,
                "predictions_by_epoch_npy": pred_path,
                "sq_error_by_epoch_npy": sq_path,
                "abs_error_by_epoch_npy": abs_path,
                "audit_epoch_metrics_json": metrics_path,
            },
            "formula_features": teacher_summary["formula_features"],
            "formula_config": teacher_summary["formula_config"],
            "global_stats": teacher_summary["global_stats"],
            "source_stats": teacher_summary["source_stats"],
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("[audit:%s] Saved teacher cache -> %s", self.tag, audit_path)
        logger.info("[audit:%s] Saved teacher summary -> %s", self.tag, summary_path)
        return summary
