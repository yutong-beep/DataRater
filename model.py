# model.py
import random
import logging
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel
from torch.func import functional_call

logger = logging.getLogger(__name__)

ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
ESM_HIDDEN = 320  # esm2_t6_8M hidden size


# =========================
# 1) Backbone + Regression Head
# =========================
class ESMForAffinity(nn.Module):
    """
    ESM2 encoder + mean pooling + MLP regressor.
    - Caches initial state for fast reset (no repeated from_pretrained()).
    - Optional: force eager attention to avoid SDPA/flash kernels.
    """
    def __init__(
        self,
        model_name: str = ESM_MODEL_NAME,
        cache_init_state: bool = True,
        force_eager_attn: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.force_eager_attn = force_eager_attn

        # Load once (HF caches locally after first download).
        if force_eager_attn:
            # Hard-disable SDPA/flash path at the transformers level.
            self.esm = EsmModel.from_pretrained(model_name, attn_implementation="eager")
        else:
            self.esm = EsmModel.from_pretrained(model_name)

        self.mlp = nn.Sequential(
            nn.Linear(ESM_HIDDEN, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self._init_state = None
        if cache_init_state:
            self._init_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, L, H]

        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # [B, L, 1]
        summed = torch.sum(hidden * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = summed / denom  # [B, H]

        pred = self.mlp(pooled).squeeze(-1)  # [B]
        return pred

    @torch.no_grad()
    def reset_parameters(self):
        """
        Fast reset without touching HF Hub.
        Restores initial snapshot if available; otherwise resets linear layers only.
        """
        if self._init_state is not None:
            self.load_state_dict(self._init_state, strict=True)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()

    def get_trainable_params(self):
        return list(self.parameters())


def functional_forward(model: nn.Module, params: Dict[str, torch.Tensor], input_ids, attention_mask) -> torch.Tensor:
    return functional_call(model, params, (input_ids, attention_mask)).squeeze(-1)


# =========================
# 2) Meta-training
# =========================
def _safe_mean(grads: List[Optional[torch.Tensor]], like: torch.Tensor) -> torch.Tensor:
    valid = [g for g in grads if g is not None]
    if not valid:
        return torch.zeros_like(like)
    return torch.stack(valid, dim=0).mean(dim=0)


def _canary_probe(
    data_rater: nn.Module,
    train_dataset,
    train_raw,
    n_probe: int = 500,
    device: Optional[torch.device] = None,
) -> dict:
    """Score a random subset and compute diagnostic metrics."""
    import random as _rnd
    import numpy as _np
    from scipy.stats import spearmanr as _sp

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_rater.eval()
    indices = _rnd.sample(range(len(train_dataset)), min(n_probe, len(train_dataset)))
    scores: List[float] = []
    pkds: List[float] = []
    sources: List[str] = []

    with torch.no_grad():
        for idx in indices:
            sample = train_dataset[idx]
            ids = sample["input_ids"].unsqueeze(0).to(device)
            mask = sample["attention_mask"].unsqueeze(0).to(device)
            s = data_rater(ids, mask).item()
            scores.append(float(s))
            pkds.append(float(sample["affinity"]))

            raw_v = sample.get("raw_index", -1)
            raw_idx = int(raw_v.item()) if torch.is_tensor(raw_v) else int(raw_v)
            if train_raw is not None and "source" in train_raw.column_names and 0 <= raw_idx < len(train_raw):
                sources.append(str(train_raw[raw_idx]["source"]))
            else:
                sources.append("UNKNOWN")

    data_rater.train()

    scores_arr = _np.array(scores, dtype=float)
    pkds_arr = _np.array(pkds, dtype=float)
    sources_arr = _np.array(sources, dtype=object)

    try:
        global_spearman = float(_sp(scores_arr, pkds_arr).correlation)
    except Exception:
        global_spearman = float("nan")

    source_iqrs: Dict[str, float] = {}
    for src in _np.unique(sources_arr):
        src_scores = scores_arr[sources_arr == src]
        if len(src_scores) > 4:
            source_iqrs[src] = float(_np.percentile(src_scores, 75) - _np.percentile(src_scores, 25))

    return {"spearman": global_spearman, "source_iqrs": source_iqrs}


def _pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    pred_c = pred - pred.mean()
    target_c = target - target.mean()
    denom = pred_c.norm(p=2) * target_c.norm(p=2) + eps
    rho = torch.sum(pred_c * target_c) / denom
    return 1.0 - rho


def _cosine_loss(pred: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    pred_c = pred - pred.mean()
    target_c = target - target.mean()
    denom = pred_c.norm(p=2) * target_c.norm(p=2) + eps
    cos = torch.sum(pred_c * target_c) / denom
    return 1.0 - cos


def _build_source_std_stats(
    train_raw,
    clamp_min: float = 1e-3,
) -> Tuple[Dict[str, float], float]:
    src2vals: Dict[str, List[float]] = {}
    all_vals: List[float] = []
    has_source = "source" in train_raw.column_names
    has_pkd = "pkd" in train_raw.column_names
    if not has_pkd:
        raise ValueError("train_raw must contain 'pkd' for source-normalized objectives.")

    sources = train_raw["source"] if has_source else ["__global__"] * len(train_raw)
    targets = train_raw["pkd"]

    for src, y in zip(sources, targets):
        try:
            fy = float(y)
        except (TypeError, ValueError):
            continue
        if not torch.isfinite(torch.tensor(fy)):
            continue
        key = str(src)
        src2vals.setdefault(key, []).append(fy)
        all_vals.append(fy)

    if not all_vals:
        global_std = 1.0
        return {}, global_std

    global_std = float(torch.tensor(all_vals, dtype=torch.float32).std(unbiased=False).item())
    global_std = max(global_std, clamp_min)

    src2std: Dict[str, float] = {}
    for src, vals in src2vals.items():
        if len(vals) < 2:
            src_std = global_std
        else:
            src_std = float(torch.tensor(vals, dtype=torch.float32).std(unbiased=False).item())
        src2std[src] = max(src_std, clamp_min)

    return src2std, global_std


# ---- v2: per-source z-score normalization stats ----
def _build_source_zscore_stats(
    raw_dataset,
    clamp_min: float = 1e-3,
) -> Tuple[Dict[str, Tuple[float, float]], float, float]:
    """
    Build per-source (mean, std) for z-score normalization of pkd targets.
    Returns: (src2stats, global_mean, global_std)
    """
    src2vals: Dict[str, List[float]] = {}
    all_vals: List[float] = []
    has_source = "source" in raw_dataset.column_names
    sources = raw_dataset["source"] if has_source else ["__global__"] * len(raw_dataset)
    targets = raw_dataset["pkd"]

    for src, y in zip(sources, targets):
        try:
            fy = float(y)
        except (TypeError, ValueError):
            continue
        if not torch.isfinite(torch.tensor(fy)):
            continue
        src2vals.setdefault(str(src), []).append(fy)
        all_vals.append(fy)

    global_mean = float(torch.tensor(all_vals).mean().item()) if all_vals else 0.0
    global_std = max(float(torch.tensor(all_vals).std(unbiased=False).item()), clamp_min) if all_vals else 1.0

    src2stats: Dict[str, Tuple[float, float]] = {}
    for src, vals in src2vals.items():
        t = torch.tensor(vals)
        m = float(t.mean().item())
        s = max(float(t.std(unbiased=False).item()), clamp_min) if len(vals) >= 2 else global_std
        src2stats[src] = (m, s)

    return src2stats, global_mean, global_std


def _zscore_normalize_targets(
    targets: torch.Tensor,
    raw_indices: torch.Tensor,
    raw_dataset,
    src2stats: Dict[str, Tuple[float, float]],
    global_mean: float,
    global_std: float,
) -> torch.Tensor:
    """
    Apply per-source z-score: z = (pkd - src_mean) / src_std.
    Used ONLY during Phase 2 inner loop.
    """
    device = targets.device
    means = []
    stds = []
    n_raw = len(raw_dataset)
    has_source = "source" in raw_dataset.column_names

    for idx in raw_indices.detach().cpu().tolist():
        if 0 <= int(idx) < n_raw and has_source:
            src = str(raw_dataset[int(idx)]["source"])
            m, s = src2stats.get(src, (global_mean, global_std))
        else:
            m, s = global_mean, global_std
        means.append(m)
        stds.append(s)

    means_t = torch.tensor(means, dtype=torch.float32, device=device)
    stds_t = torch.tensor(stds, dtype=torch.float32, device=device)
    return (targets - means_t) / stds_t


def _lookup_batch_src_std(
    raw_indices: torch.Tensor,
    val_raw,
    src2std: Dict[str, float],
    global_std: float,
    device: torch.device,
) -> torch.Tensor:
    vals: List[float] = []
    n_val = len(val_raw)
    has_source = "source" in val_raw.column_names
    for idx in raw_indices.detach().cpu().tolist():
        src_std = global_std
        if 0 <= int(idx) < n_val and has_source:
            src = str(val_raw[int(idx)]["source"])
            src_std = src2std.get(src, global_std)
        vals.append(float(src_std))
    return torch.tensor(vals, dtype=torch.float32, device=device)


def _compute_outer_loss(
    objective: str,
    preds: torch.Tensor,
    targets: torch.Tensor,
    src_std: Optional[torch.Tensor],
    alpha: float,
    outer_eps: float,
    mse_norm_eps: float,
    source_labels: Optional[List[str]] = None,
) -> torch.Tensor:
    if objective == "pearson":
        return _pearson_loss(preds, targets, outer_eps)
    if objective == "cosine":
        return _cosine_loss(preds, targets, outer_eps)

    # ---- v2: source-stratified MSE ----
    if objective == "source_stratified_mse":
        if source_labels is None:
            return F.mse_loss(preds, targets)
        per_sample_mse = (preds - targets) ** 2
        unique_sources = list(set(source_labels))
        source_losses = []
        for src in unique_sources:
            mask = torch.tensor([s == src for s in source_labels], dtype=torch.bool, device=preds.device)
            if mask.sum() > 0:
                source_losses.append(per_sample_mse[mask].mean())
        if not source_losses:
            return F.mse_loss(preds, targets)
        return torch.stack(source_losses).mean()

    if src_std is None:
        src_std = torch.ones_like(targets)
    mse_norm = torch.mean((preds - targets) ** 2 / (src_std ** 2 + mse_norm_eps))
    if objective == "mse_norm":
        return mse_norm
    if objective == "mix":
        pearson_term = _pearson_loss(preds, targets, outer_eps)
        return alpha * pearson_term + (1.0 - alpha) * mse_norm

    raise ValueError(f"Unsupported outer_objective: {objective}")


def train_datarater(
    train_loader,
    val_loader,
    n_meta_steps: int = 5000,
    n_inner_models: int = 8,
    lifetime: int = 2000,
    T_window: int = 2,
    T_backprop: int = 2,
    use_first_order_ablation: bool = False,
    sample_one_inner: bool = False,
    inner_lr: float = 1e-4,
    tau: float = 0.5,
    outer_objective: str = "mse_norm",
    alpha: float = 0.5,
    outer_eps: float = 1e-8,
    mse_norm_eps: float = 1e-6,
    train_raw=None,
    val_raw=None,
    device: Optional[torch.device] = None,
    force_eager_attn: bool = False,  # set True if you still see SDPA issues
    use_zscore_inner: bool = False,
    meta_grad_clip: float = 1.0,
    train_dataset=None,
    canary_interval: int = 200,
) -> ESMForAffinity:
    """
    Meta-learn DataRater weights for samples.

    Critical fix for 2nd-order gradients:
    - force math SDPA inside this function via sdp_kernel context.
    """

    # ---- (A) Best practice: only affect meta-training, not baseline ----
    from torch.backends.cuda import sdp_kernel
    sdp_ctx = sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    sdp_ctx.__enter__()
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        allowed_objectives = {"mse_norm", "pearson", "cosine", "mix", "source_stratified_mse"}
        if outer_objective not in allowed_objectives:
            raise ValueError(f"outer_objective must be one of {sorted(allowed_objectives)}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")

        # v2: clamp T_backprop
        T_backprop_eff = min(T_backprop, T_window)
        T_warmup = T_window - T_backprop_eff  # Warmup steps are detached (no graph construction)
        logger.info(f"[meta-v2] T_window={T_window}, T_backprop={T_backprop_eff}, T_warmup={T_warmup}")

        src2std: Dict[str, float] = {}
        global_std = 1.0
        if outer_objective in {"mse_norm", "mix"}:
            if train_raw is None:
                raise ValueError("train_raw is required for outer_objective in {'mse_norm', 'mix'}.")
            if val_raw is None:
                raise ValueError("val_raw is required for outer_objective in {'mse_norm', 'mix'}.")
            src2std, global_std = _build_source_std_stats(train_raw, clamp_min=1e-3)
            logger.info(
                "[meta] source-std stats ready | num_sources=%d | global_std=%.6f",
                len(src2std),
                global_std,
            )

        # v2: per-source z-score stats
        src2zscore: Dict[str, Tuple[float, float]] = {}
        zscore_global_mean, zscore_global_std = 0.0, 1.0
        if use_zscore_inner and train_raw is not None:
            src2zscore, zscore_global_mean, zscore_global_std = _build_source_zscore_stats(train_raw)
            logger.info(
                "[meta-v2] z-score stats ready | num_sources=%d | global_mean=%.4f | global_std=%.4f",
                len(src2zscore), zscore_global_mean, zscore_global_std,
            )
            for src, (m, s) in src2zscore.items():
                logger.info(f"  {src}: mean={m:.4f}, std={s:.4f}")

        # DataRater (heavy)
        data_rater = ESMForAffinity(cache_init_state=True, force_eager_attn=force_eager_attn).to(device)
        rater_opt = torch.optim.Adam(data_rater.parameters(), lr=1e-4)
        rater_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            rater_opt, T_max=n_meta_steps, eta_min=1e-6
        )

        # Inner population (heavy)
        population = [
            ESMForAffinity(cache_init_state=True, force_eager_attn=force_eager_attn).to(device)
            for _ in range(n_inner_models)
        ]

        train_iter = iter(train_loader)
        val_iter = iter(val_loader)

        data_rater.train()

        for step in range(n_meta_steps):
            # staggered resets
            for i, inner_model in enumerate(population):
                offset = (n_inner_models - 1 - i) * (lifetime // n_inner_models)
                if step > 0 and (step + offset) % lifetime == 0:
                    inner_model.reset_parameters()

            models_to_process = (
                [random.randint(0, n_inner_models - 1)] if sample_one_inner else list(range(n_inner_models))
            )

            rater_opt.zero_grad(set_to_none=True)
            meta_grads_accumulator = []

            for m_idx in models_to_process:
                inner_model = population[m_idx]
                fast_weights = dict(inner_model.named_parameters())

                # ---- inner loop (v2: K-step truncated BPTT) ----
                for _t in range(T_window):
                    try:
                        x_in = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        x_in = next(train_iter)

                    input_ids = x_in["input_ids"].to(device)
                    mask = x_in["attention_mask"].to(device)
                    targets = x_in["affinity"].to(device)

                    # v2: per-source z-score normalization of inner targets
                    if use_zscore_inner and src2zscore and "raw_index" in x_in:
                        targets = _zscore_normalize_targets(
                            targets, x_in["raw_index"].to(device),
                            train_raw, src2zscore,
                            zscore_global_mean, zscore_global_std,
                        )

                    if _t == T_warmup:
                        fast_weights = {
                            k: v.detach().requires_grad_(True)
                            for k, v in fast_weights.items()
                        }

                    # v2: only build graph for the last T_backprop steps
                    is_graph_step = (_t >= T_warmup)

                    with torch.set_grad_enabled(is_graph_step):
                        raw_scores = data_rater(input_ids, mask)      # [B]
                        weights = F.softmax(raw_scores / tau, dim=0)  # [B]

                    preds = functional_forward(inner_model, fast_weights, input_ids, mask)
                    per = F.mse_loss(preds, targets, reduction="none")
                    inner_loss = torch.sum(weights * per)

                    need_graph = is_graph_step and (not use_first_order_ablation)
                    grads = torch.autograd.grad(
                        inner_loss,
                        tuple(fast_weights.values()),
                        create_graph=need_graph,
                        allow_unused=True,
                    )

                    if not need_graph:
                        grads = [g.detach() if g is not None else None for g in grads]

                    fast_weights = {
                        name: (w - inner_lr * g) if g is not None else w
                        for (name, w), g in zip(fast_weights.items(), grads)
                    }

                # ---- outer loop ----
                try:
                    x_out = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    x_out = next(val_iter)

                out_ids = x_out["input_ids"].to(device)
                out_mask = x_out["attention_mask"].to(device)
                out_targets = x_out["affinity"].to(device)
                out_raw_index = x_out.get("raw_index")
                if out_raw_index is not None:
                    out_raw_index = out_raw_index.to(device)

                outer_preds = functional_forward(inner_model, fast_weights, out_ids, out_mask)
                batch_src_std = None
                if outer_objective in {"mse_norm", "mix"}:
                    if out_raw_index is not None and val_raw is not None:
                        batch_src_std = _lookup_batch_src_std(out_raw_index, val_raw, src2std, global_std, device)
                    else:
                        batch_src_std = torch.full_like(out_targets, fill_value=float(global_std))

                # v2: get source labels for stratified outer loss
                batch_source_labels = None
                if outer_objective == "source_stratified_mse" and out_raw_index is not None and val_raw is not None:
                    n_val = len(val_raw)
                    has_src = "source" in val_raw.column_names
                    batch_source_labels = []
                    for idx in out_raw_index.detach().cpu().tolist():
                        if 0 <= int(idx) < n_val and has_src:
                            batch_source_labels.append(str(val_raw[int(idx)]["source"]))
                        else:
                            batch_source_labels.append("__unknown__")

                outer_loss = _compute_outer_loss(
                    objective=outer_objective,
                    preds=outer_preds,
                    targets=out_targets,
                    src_std=batch_src_std,
                    alpha=alpha,
                    outer_eps=outer_eps,
                    mse_norm_eps=mse_norm_eps,
                    source_labels=batch_source_labels,
                )

                if use_first_order_ablation:
                    # Keep direct path to DataRater in first-order mode.
                    out_scores = data_rater(out_ids, out_mask)
                    out_w = F.softmax(out_scores / tau, dim=0)
                    outer_loss = outer_loss + 0.01 * torch.sum(out_w * (outer_preds.detach() - out_targets).pow(2))

                meta_grads = torch.autograd.grad(
                    outer_loss,
                    tuple(data_rater.parameters()),
                    allow_unused=True,
                )
                meta_grads_accumulator.append(meta_grads)

                # truncate and sync
                with torch.no_grad():
                    for name, p in inner_model.named_parameters():
                        p.copy_(fast_weights[name].detach())

            # ---- apply averaged grads ----
            params = list(data_rater.parameters())
            for p, grads_for_p in zip(params, zip(*meta_grads_accumulator)):
                p.grad = _safe_mean(list(grads_for_p), p)

            torch.nn.utils.clip_grad_norm_(data_rater.parameters(), max_norm=float(meta_grad_clip))
            rater_opt.step()
            rater_scheduler.step()

            if (step + 1) % 50 == 0:
                with torch.no_grad():
                    gsum = 0.0
                    for p in data_rater.parameters():
                        gsum += float(p.grad.norm().item()) if p.grad is not None else 0.0
                    outer_loss_value = float(outer_loss.detach().item())
                    current_lr = float(rater_scheduler.get_last_lr()[0])
                    msg = (
                        f"[meta] step {step+1}/{n_meta_steps} | obj={outer_objective} "
                        f"| outer_loss={outer_loss_value:.6f} | grad_norm_sum={gsum:.4f} "
                        f"| lr={current_lr:.2e}"
                    )
                    if batch_src_std is not None:
                        msg += f" | mean_src_std={float(batch_src_std.mean().item()):.6f}"
                logger.info(msg)

            if train_dataset is not None and canary_interval > 0 and (step + 1) % canary_interval == 0:
                canary = _canary_probe(data_rater, train_dataset, train_raw, n_probe=500, device=device)
                canary_msg = f"[canary] step {step+1} | Spearman={canary['spearman']:.4f}"
                for src, iqr_val in sorted(canary["source_iqrs"].items()):
                    canary_msg += f" | {src}_IQR={iqr_val:.3f}"
                logger.info(canary_msg)

                pdz_iqr = canary["source_iqrs"].get("PDZ_PBM")
                if pdz_iqr is not None and pdz_iqr < 1.0:
                    logger.warning(
                        f"[CANARY WARNING] step {step+1}: PDZ_PBM IQR={pdz_iqr:.4f}"
                        f" < 1.0 -- DataRater may be collapsing!"
                    )

        return data_rater

    finally:
        # always restore SDPA backend state
        sdp_ctx.__exit__(None, None, None)


# =========================
# 3) Filtering (non-differentiable by design)
# =========================
def filter_dataset(data_rater, original_dataset, N_ref=10000, B=256, keep_ratio=0.7, return_indices: bool = False):
    import bisect
    from scipy.stats import binom

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_rater.eval()

    print(f"Building Empirical CDF using {N_ref} random samples...")
    indices = random.sample(range(len(original_dataset)), min(N_ref, len(original_dataset)))
    ref_scores = []

    with torch.no_grad():
        for idx in indices:
            sample = original_dataset[idx]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            mask = sample["attention_mask"].unsqueeze(0).to(device)
            score = data_rater(input_ids, mask).item()
            ref_scores.append(score)

    ref_scores.sort()
    K = int(B * keep_ratio)
    filtered_indices = []

    print("Filtering dataset using P_accept formula...")
    with torch.no_grad():
        for idx in range(len(original_dataset)):
            sample = original_dataset[idx]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            mask = sample["attention_mask"].unsqueeze(0).to(device)

            score = data_rater(input_ids, mask).item()
            pos = bisect.bisect_left(ref_scores, score)
            p = pos / max(1, len(ref_scores))

            p_accept = binom.cdf(K - 1, B - 1, 1 - p)
            if random.random() < p_accept:
                filtered_indices.append(idx)

    filtered_dataset = original_dataset.select(filtered_indices)
    print(f"Original size: {len(original_dataset)}, Filtered size: {len(filtered_dataset)}")
    if return_indices:
        return filtered_dataset, filtered_indices
    return filtered_dataset
