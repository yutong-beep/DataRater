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
) -> torch.Tensor:
    if objective == "pearson":
        return _pearson_loss(preds, targets, outer_eps)
    if objective == "cosine":
        return _cosine_loss(preds, targets, outer_eps)

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
        allowed_objectives = {"mse_norm", "pearson", "cosine", "mix"}
        if outer_objective not in allowed_objectives:
            raise ValueError(f"outer_objective must be one of {sorted(allowed_objectives)}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")

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

        # DataRater (heavy)
        data_rater = ESMForAffinity(cache_init_state=True, force_eager_attn=force_eager_attn).to(device)
        rater_opt = torch.optim.Adam(data_rater.parameters(), lr=1e-4)

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

                # ---- inner loop ----
                for _t in range(T_window):
                    try:
                        x_in = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        x_in = next(train_iter)

                    input_ids = x_in["input_ids"].to(device)
                    mask = x_in["attention_mask"].to(device)
                    targets = x_in["affinity"].to(device)

                    raw_scores = data_rater(input_ids, mask)      # [B]
                    weights = F.softmax(raw_scores / tau, dim=0)  # [B]

                    preds = functional_forward(inner_model, fast_weights, input_ids, mask)
                    per = F.mse_loss(preds, targets, reduction="none")
                    inner_loss = torch.sum(weights * per)

                    grads = torch.autograd.grad(
                        inner_loss,
                        tuple(fast_weights.values()),
                        create_graph=not use_first_order_ablation,
                        allow_unused=True,
                    )

                    if use_first_order_ablation:
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
                outer_loss = _compute_outer_loss(
                    objective=outer_objective,
                    preds=outer_preds,
                    targets=out_targets,
                    src_std=batch_src_std,
                    alpha=alpha,
                    outer_eps=outer_eps,
                    mse_norm_eps=mse_norm_eps,
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

            rater_opt.step()

            if (step + 1) % 50 == 0:
                with torch.no_grad():
                    gsum = 0.0
                    for p in data_rater.parameters():
                        gsum += float(p.grad.norm().item()) if p.grad is not None else 0.0
                    outer_loss_value = float(outer_loss.detach().item())
                    msg = (
                        f"[meta] step {step+1}/{n_meta_steps} | obj={outer_objective} "
                        f"| outer_loss={outer_loss_value:.6f} | grad_norm_sum={gsum:.4f}"
                    )
                    if batch_src_std is not None:
                        msg += f" | mean_src_std={float(batch_src_std.mean().item()):.6f}"
                logger.info(msg)

        return data_rater

    finally:
        # always restore SDPA backend state
        sdp_ctx.__exit__(None, None, None)


# =========================
# 3) Filtering (non-differentiable by design)
# =========================
def filter_dataset(data_rater, original_dataset, N_ref=10000, B=256, keep_ratio=0.7):
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
    return filtered_dataset
