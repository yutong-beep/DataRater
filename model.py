# model.py
import math
import random
import logging
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel
from torch.func import functional_call

logger = logging.getLogger(__name__)

# =========================
# Backbone + Regressor
# =========================
ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
ESM_HIDDEN = 320  # esm2_t6_8M hidden size


class ESMForAffinity(nn.Module):
    """
    ESM2 encoder + mean pooling + MLP regressor.
    Includes fast reset without re-downloading weights.
    """
    def __init__(self, model_name: str = ESM_MODEL_NAME, cache_init_state: bool = True):
        super().__init__()
        self.model_name = model_name

        # Load once. (HF will cache locally after first download.)
        self.esm = EsmModel.from_pretrained(model_name)
        self.mlp = nn.Sequential(
            nn.Linear(ESM_HIDDEN, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self._init_state = None
        if cache_init_state:
            # Store an initial snapshot on CPU (so resets are cheap + no HF calls).
            self._init_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, L, H]

        # mean pooling with mask
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # [B, L, 1]
        summed = torch.sum(hidden * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = summed / denom  # [B, H]

        pred = self.mlp(pooled).squeeze(-1)  # [B]
        return pred

    @torch.no_grad()
    def reset_parameters(self):
        """
        Fast reset: restore to cached init weights if available.
        No from_pretrained() here -> no repeated HF checks/downloads.
        """
        if self._init_state is None:
            # Fallback: reset only linear layers (still no HF calls)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
            return

        self.load_state_dict(self._init_state, strict=True)

    def get_trainable_params(self):
        return list(self.parameters())


def functional_forward(model: nn.Module, params: Dict[str, torch.Tensor], input_ids, attention_mask) -> torch.Tensor:
    """
    Forward pass using explicit parameter dict (fast_weights).
    Keeps computation graph for higher-order grads.
    """
    return functional_call(model, params, (input_ids, attention_mask)).squeeze(-1)


# =========================
# Meta-training DataRater
# =========================
def _safe_mean(grads: List[Optional[torch.Tensor]], like: torch.Tensor) -> torch.Tensor:
    """
    Average gradients, treating None as zero. Returns a tensor with same shape as `like`.
    """
    valid = [g for g in grads if g is not None]
    if not valid:
        return torch.zeros_like(like)
    return torch.stack(valid, dim=0).mean(dim=0)


def train_datarater(
    train_loader,
    val_loader,
    n_meta_steps: int = 5000,
    n_inner_models: int = 8,
    lifetime: int = 2000,
    T_window: int = 2,
    use_first_order_ablation: bool = False,   # your ablation flag
    sample_one_inner: bool = False,
    inner_lr: float = 1e-4,
    tau: float = 0.5,
    device: Optional[torch.device] = None,
) -> ESMForAffinity:
    """
    Meta-learn a DataRater that assigns per-sample weights.

    - Inner loop uses data_rater weights for weighted MSE.
    - Outer loop computes validation loss.
    - Ablation mode: do NOT backprop through unrolled inner updates,
      but still allow DataRater to learn via a direct outer weighting path.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataRater (heavy, but at least not repeatedly reloaded)
    data_rater = ESMForAffinity(cache_init_state=True).to(device)
    rater_opt = torch.optim.Adam(data_rater.parameters(), lr=1e-4)

    # Inner population (each cached for fast reset)
    population = [ESMForAffinity(cache_init_state=True).to(device) for _ in range(n_inner_models)]

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    data_rater.train()

    for step in range(n_meta_steps):
        # Staggered resets (no HF calls now)
        for i, inner_model in enumerate(population):
            offset = (n_inner_models - 1 - i) * (lifetime // n_inner_models)
            if step > 0 and (step + offset) % lifetime == 0:
                inner_model.reset_parameters()

        if sample_one_inner:
            models_to_process = [random.randint(0, n_inner_models - 1)]
        else:
            models_to_process = list(range(n_inner_models))

        rater_opt.zero_grad(set_to_none=True)

        meta_grads_accumulator = []

        for m_idx in models_to_process:
            inner_model = population[m_idx]
            fast_weights = dict(inner_model.named_parameters())

            # -------- Inner loop (T steps) --------
            for _t in range(T_window):
                try:
                    x_in = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x_in = next(train_iter)

                input_ids = x_in["input_ids"].to(device)
                mask = x_in["attention_mask"].to(device)
                targets = x_in["affinity"].to(device)

                # DataRater scores and differentiable weights
                raw_scores = data_rater(input_ids, mask)              # [B]
                weights = F.softmax(raw_scores / tau, dim=0)          # [B]

                preds = functional_forward(inner_model, fast_weights, input_ids, mask)
                per = F.mse_loss(preds, targets, reduction="none")    # [B]
                inner_loss = torch.sum(weights * per)                 # scalar

                grads = torch.autograd.grad(
                    inner_loss,
                    tuple(fast_weights.values()),
                    create_graph=not use_first_order_ablation,
                    allow_unused=True,
                )

                if use_first_order_ablation:
                    # stop gradients through the unrolled states (fast_weights trajectory)
                    grads = [g.detach() if g is not None else None for g in grads]

                fast_weights = {
                    name: (w - inner_lr * g) if g is not None else w
                    for (name, w), g in zip(fast_weights.items(), grads)
                }

            # -------- Outer loop --------
            try:
                x_out = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                x_out = next(val_iter)

            out_ids = x_out["input_ids"].to(device)
            out_mask = x_out["attention_mask"].to(device)
            out_targets = x_out["affinity"].to(device)

            outer_preds = functional_forward(inner_model, fast_weights, out_ids, out_mask)
            outer_per = F.mse_loss(outer_preds, out_targets, reduction="none")

            if use_first_order_ablation:
                # Key: Make outer loss directly depend on DataRater (so it can still learn)
                out_scores = data_rater(out_ids, out_mask)
                out_w = F.softmax(out_scores / tau, dim=0)
                outer_loss = torch.sum(out_w * outer_per)
            else:
                outer_loss = outer_per.mean()

            # meta-grad wrt DataRater params (never crash; None means truly disconnected)
            meta_grads = torch.autograd.grad(
                outer_loss,
                tuple(data_rater.parameters()),
                allow_unused=True,
            )
            meta_grads_accumulator.append(meta_grads)

            # Sync back to the actual inner model (truncate graph)
            with torch.no_grad():
                for name, p in inner_model.named_parameters():
                    p.copy_(fast_weights[name].detach())

        # -------- Update DataRater: average per-param grads --------
        params = list(data_rater.parameters())
        # transpose list-of-tuples: [(g1p1,g1p2,...), (g2p1,g2p2,...)] -> per param
        for p, grads_for_p in zip(params, zip(*meta_grads_accumulator)):
            p.grad = _safe_mean(list(grads_for_p), p)

        rater_opt.step()

        if (step + 1) % 50 == 0:
            # light logging
            gnorm = 0.0
            with torch.no_grad():
                for p in data_rater.parameters():
                    if p.grad is not None:
                        gnorm += float(p.grad.norm().item())
            logger.info(f"[meta] step {step+1}/{n_meta_steps} | grad_norm_sum={gnorm:.4f}")

    return data_rater


# =========================
# Dataset filtering (kept as-is, but note: uses no_grad and .item() on purpose)
# =========================
def filter_dataset(data_rater, original_dataset, N_ref=10000, B=256, keep_ratio=0.7):
    """
    NOTE: This is a non-differentiable filtering stage by design (uses .item()).
    It should be used AFTER meta-training, not inside the differentiable loop.
    """
    import bisect
    import numpy as np
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