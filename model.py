# model.py
import math
import random
import logging
from typing import Dict, List, Tuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel
from torch.func import functional_call

from mixflow_mg import mixflow_inner_update

logger = logging.getLogger(__name__)

ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
ESM_HIDDEN = 320  # esm2_t6_8M hidden size
DEFAULT_HARD_OUTER_SOURCES = ["SKEMPI v2.0", "PDBbind v2020"]


def _mean_pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = torch.sum(hidden * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


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

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, L, H]
        return _mean_pool_hidden(hidden, attention_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(input_ids, attention_mask)
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


class MultiHeadDataRater(nn.Module):
    """
    Shared ESM2 trunk plus one score head per source.
    """
    def __init__(
        self,
        source_names: Sequence[str],
        model_name: str = ESM_MODEL_NAME,
        cache_init_state: bool = True,
        force_eager_attn: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.force_eager_attn = force_eager_attn
        self.source_names = [str(s) for s in source_names]
        if not self.source_names:
            raise ValueError("MultiHeadDataRater requires at least one source name.")
        self.source_to_head_key = {
            src: f"source_{idx}"
            for idx, src in enumerate(self.source_names)
        }

        if force_eager_attn:
            self.esm = EsmModel.from_pretrained(model_name, attn_implementation="eager")
        else:
            self.esm = EsmModel.from_pretrained(model_name)

        self.heads = nn.ModuleDict({
            self.source_to_head_key[src]: nn.Sequential(
                nn.Linear(ESM_HIDDEN, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            for src in self.source_names
        })

        self._init_state = None
        if cache_init_state:
            self._init_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        return _mean_pool_hidden(hidden, attention_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, sources: Sequence[str]) -> torch.Tensor:
        if len(sources) != int(input_ids.size(0)):
            raise ValueError(f"Expected {input_ids.size(0)} sources, got {len(sources)}")

        pooled = self.encode(input_ids, attention_mask)
        scores = torch.empty(pooled.size(0), dtype=pooled.dtype, device=pooled.device)
        grouped: Dict[str, List[int]] = {}
        for idx, source_name in enumerate(sources):
            source_key = str(source_name)
            head_key = self.source_to_head_key.get(source_key)
            if head_key is None:
                raise KeyError(f"Unknown source for MultiHeadDataRater: {source_key}")
            grouped.setdefault(head_key, []).append(idx)

        for head_key, idxs in grouped.items():
            idx_t = torch.tensor(idxs, dtype=torch.long, device=pooled.device)
            scores[idx_t] = self.heads[head_key](pooled[idx_t]).squeeze(-1)
        return scores

    @torch.no_grad()
    def reset_parameters(self):
        if self._init_state is not None:
            self.load_state_dict(self._init_state, strict=True)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()

    def get_trainable_params(self):
        return list(self.parameters())


class MoEDataRater(nn.Module):
    """
    Learned sparse Mixture-of-Experts scorer for Phase-2 DataRater.
    """
    def __init__(
        self,
        model_name: str = ESM_MODEL_NAME,
        num_experts: int = 4,
        router_top_k: int = 2,
        capacity_factor: float = 1.25,
        router_temperature: float = 1.0,
        router_noise_std: float = 0.0,
        moe_score_merge: str = "weighted_sum",
        drop_overflow_tokens: bool = True,
        cache_init_state: bool = True,
        force_eager_attn: bool = False,
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        if router_top_k not in {1, 2}:
            raise ValueError("router_top_k must be 1 or 2.")
        if router_top_k > num_experts:
            raise ValueError("router_top_k cannot exceed num_experts.")
        if capacity_factor <= 0.0:
            raise ValueError("capacity_factor must be positive.")
        if router_temperature <= 0.0:
            raise ValueError("router_temperature must be positive.")
        if moe_score_merge not in {"weighted_sum", "top1_only"}:
            raise ValueError("moe_score_merge must be one of {'weighted_sum', 'top1_only'}.")

        self.model_name = model_name
        self.force_eager_attn = force_eager_attn
        self.num_experts = int(num_experts)
        self.router_top_k = int(router_top_k)
        self.capacity_factor = float(capacity_factor)
        self.router_temperature = float(router_temperature)
        self.router_noise_std = float(router_noise_std)
        self.moe_score_merge = str(moe_score_merge)
        self.drop_overflow_tokens = bool(drop_overflow_tokens)

        if force_eager_attn:
            self.esm = EsmModel.from_pretrained(model_name, attn_implementation="eager")
        else:
            self.esm = EsmModel.from_pretrained(model_name)

        self.router = nn.Linear(ESM_HIDDEN, self.num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ESM_HIDDEN, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            for _ in range(self.num_experts)
        ])

        self._init_state = None
        if cache_init_state:
            self._init_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        return _mean_pool_hidden(hidden, attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_router_info: bool = False,
    ):
        pooled = self.encode(input_ids, attention_mask)
        batch_size = int(pooled.size(0))
        device = pooled.device
        dtype = pooled.dtype

        router_logits = self.router(pooled) / self.router_temperature
        if self.training and self.router_noise_std > 0.0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_noise_std

        router_probs = F.softmax(router_logits, dim=-1)
        topk_vals, topk_idx = torch.topk(router_probs, k=self.router_top_k, dim=-1)
        topk_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        capacity = max(1, int(math.ceil(self.capacity_factor * batch_size * self.router_top_k / self.num_experts)))
        accepted_mask = torch.zeros((batch_size, self.router_top_k), dtype=torch.bool, device=device)
        route_scores = torch.zeros((batch_size, self.router_top_k), dtype=dtype, device=device)
        expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        overflow_count = torch.tensor(0, dtype=torch.long, device=device)

        assignments = []
        for token_idx in range(batch_size):
            for route_pos in range(self.router_top_k):
                assignments.append(
                    (
                        float(topk_vals[token_idx, route_pos].detach().item()),
                        int(token_idx),
                        int(route_pos),
                        int(topk_idx[token_idx, route_pos].item()),
                    )
                )
        assignments.sort(key=lambda item: item[0], reverse=True)

        assignments_by_expert: Dict[int, List[Tuple[int, int]]] = {e: [] for e in range(self.num_experts)}
        for _, token_idx, route_pos, expert_idx in assignments:
            expert_is_full = int(expert_counts[expert_idx].item()) >= capacity
            if expert_is_full:
                overflow_count = overflow_count + 1
                continue

            expert_counts[expert_idx] = expert_counts[expert_idx] + 1
            accepted_mask[token_idx, route_pos] = True
            assignments_by_expert[expert_idx].append((token_idx, route_pos))

        for expert_idx, expert_assignments in assignments_by_expert.items():
            if not expert_assignments:
                continue
            token_indices = torch.tensor([token_idx for token_idx, _ in expert_assignments], dtype=torch.long, device=device)
            route_positions = torch.tensor([route_pos for _, route_pos in expert_assignments], dtype=torch.long, device=device)
            expert_outputs = self.experts[expert_idx](pooled[token_indices]).squeeze(-1)
            route_scores[token_indices, route_positions] = expert_outputs

        accepted_weights = topk_weights * accepted_mask.to(topk_weights.dtype)
        if self.moe_score_merge == "weighted_sum":
            weight_denom = accepted_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            merge_weights = accepted_weights / weight_denom
            scores = torch.sum(merge_weights * route_scores, dim=-1)
        else:
            scores = torch.zeros(batch_size, dtype=dtype, device=device)
            taken = torch.zeros(batch_size, dtype=torch.bool, device=device)
            for route_pos in range(self.router_top_k):
                route_mask = accepted_mask[:, route_pos] & (~taken)
                scores[route_mask] = route_scores[route_mask, route_pos]
                taken = taken | route_mask

        importance = router_probs.mean(dim=0)
        total_routed = expert_counts.sum().to(dtype).clamp(min=1.0)
        load = expert_counts.to(dtype) / total_routed
        aux_loss = self.num_experts * torch.sum(importance * load)
        z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)
        router_entropy = -torch.sum(router_probs * torch.log(router_probs.clamp(min=1e-9)), dim=-1).mean()
        dropped_tokens = (~accepted_mask.any(dim=-1)).sum()

        router_info = {
            "importance": importance.detach(),
            "load": load.detach(),
            "expert_counts": expert_counts.detach(),
            "overflow_count": overflow_count.detach(),
            "dropped_tokens": dropped_tokens.detach(),
            "router_entropy": router_entropy.detach(),
            "aux_loss": aux_loss,
            "z_loss": z_loss,
            "max_expert_share": load.max().detach(),
            "capacity": torch.tensor(capacity, dtype=torch.long, device=device),
        }

        if return_router_info:
            return scores, router_info
        return scores

    @torch.no_grad()
    def reset_parameters(self):
        if self._init_state is not None:
            self.load_state_dict(self._init_state, strict=True)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()

    def get_trainable_params(self):
        return list(self.parameters())


def _aggregate_router_infos(router_infos: List[Dict[str, torch.Tensor]], num_experts: int, device: torch.device):
    if not router_infos:
        return None

    importance = torch.stack([info["importance"].to(device) for info in router_infos], dim=0).mean(dim=0)
    load = torch.stack([info["load"].to(device) for info in router_infos], dim=0).mean(dim=0)
    expert_counts = torch.stack(
        [info["expert_counts"].to(device=device, dtype=torch.float32) for info in router_infos],
        dim=0,
    ).sum(dim=0)
    overflow_count = torch.stack(
        [info["overflow_count"].to(device=device, dtype=torch.float32) for info in router_infos],
        dim=0,
    ).sum()
    dropped_tokens = torch.stack(
        [info["dropped_tokens"].to(device=device, dtype=torch.float32) for info in router_infos],
        dim=0,
    ).sum()
    router_entropy = torch.stack([info["router_entropy"].to(device) for info in router_infos], dim=0).mean()
    aux_loss = torch.stack([info["aux_loss"] for info in router_infos], dim=0).mean()
    z_loss = torch.stack([info["z_loss"] for info in router_infos], dim=0).mean()
    capacity = max(int(info["capacity"].detach().item()) for info in router_infos if "capacity" in info)

    return {
        "importance": importance,
        "load": load,
        "expert_counts": expert_counts.round().to(dtype=torch.long),
        "overflow_count": overflow_count.round().to(dtype=torch.long),
        "dropped_tokens": dropped_tokens.round().to(dtype=torch.long),
        "router_entropy": router_entropy,
        "aux_loss": aux_loss,
        "z_loss": z_loss,
        "max_expert_share": load.max(),
        "capacity": torch.tensor(capacity, dtype=torch.long, device=device),
        "num_experts": int(num_experts),
    }


def infer_source_names(*raw_datasets) -> List[str]:
    names = set()
    for raw_dataset in raw_datasets:
        if raw_dataset is None or "source" not in getattr(raw_dataset, "column_names", []):
            continue
        for source_name in raw_dataset["source"]:
            if source_name is None:
                continue
            source_str = str(source_name).strip()
            if source_str:
                names.add(source_str)
    return sorted(names)


def build_datarater_model(
    arch: str = "single",
    source_names: Optional[Sequence[str]] = None,
    model_name: str = ESM_MODEL_NAME,
    cache_init_state: bool = True,
    force_eager_attn: bool = False,
    num_experts: int = 4,
    router_top_k: int = 2,
    capacity_factor: float = 1.25,
    router_temperature: float = 1.0,
    router_noise_std: float = 0.0,
    moe_score_merge: str = "weighted_sum",
    drop_overflow_tokens: bool = True,
) -> nn.Module:
    if arch == "single":
        return ESMForAffinity(
            model_name=model_name,
            cache_init_state=cache_init_state,
            force_eager_attn=force_eager_attn,
        )
    if arch == "multihead":
        uniq_sources = []
        for source_name in source_names or []:
            source_str = str(source_name).strip()
            if source_str and source_str not in uniq_sources:
                uniq_sources.append(source_str)
        if not uniq_sources:
            raise ValueError("source_names are required when datarater_arch='multihead'.")
        return MultiHeadDataRater(
            source_names=uniq_sources,
            model_name=model_name,
            cache_init_state=cache_init_state,
            force_eager_attn=force_eager_attn,
        )
    if arch == "moe":
        return MoEDataRater(
            model_name=model_name,
            num_experts=num_experts,
            router_top_k=router_top_k,
            capacity_factor=capacity_factor,
            router_temperature=router_temperature,
            router_noise_std=router_noise_std,
            moe_score_merge=moe_score_merge,
            drop_overflow_tokens=drop_overflow_tokens,
            cache_init_state=cache_init_state,
            force_eager_attn=force_eager_attn,
        )
    raise ValueError(f"Unsupported datarater arch: {arch}")


def _normalize_source_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _source_from_raw_index(raw_index: int, raw_dataset) -> str:
    if raw_dataset is None or "source" not in getattr(raw_dataset, "column_names", []):
        return "__unknown__"
    if 0 <= int(raw_index) < len(raw_dataset):
        return str(raw_dataset[int(raw_index)]["source"])
    return "__unknown__"


def _batch_sources_from_raw_indices(raw_indices, raw_dataset) -> List[str]:
    if raw_indices is None:
        raise ValueError("Multi-head DataRater requires raw_indices for source lookup.")
    if torch.is_tensor(raw_indices):
        if raw_indices.ndim == 0:
            raw_idx_list = [int(raw_indices.item())]
        else:
            raw_idx_list = [int(v) for v in raw_indices.detach().cpu().tolist()]
    elif isinstance(raw_indices, (list, tuple)):
        raw_idx_list = [int(v) for v in raw_indices]
    else:
        raw_idx_list = [int(raw_indices)]

    sources = [_source_from_raw_index(raw_idx, raw_dataset) for raw_idx in raw_idx_list]
    if any(src == "__unknown__" for src in sources):
        raise ValueError("Multi-head DataRater requires valid raw_index -> source mappings.")
    return sources


def datarater_forward(
    data_rater: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    raw_indices=None,
    raw_dataset=None,
    return_router_info: bool = False,
):
    if isinstance(data_rater, MultiHeadDataRater):
        batch_sources = _batch_sources_from_raw_indices(raw_indices, raw_dataset)
        scores = data_rater(input_ids, attention_mask, batch_sources)
        if return_router_info:
            return scores, None
        return scores
    if isinstance(data_rater, MoEDataRater):
        return data_rater(input_ids, attention_mask, return_router_info=return_router_info)
    scores = data_rater(input_ids, attention_mask)
    if return_router_info:
        return scores, None
    return scores


def functional_datarater_forward(
    data_rater: nn.Module,
    params: Dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    raw_indices=None,
    raw_dataset=None,
):
    if isinstance(data_rater, MultiHeadDataRater):
        batch_sources = _batch_sources_from_raw_indices(raw_indices, raw_dataset)
        return functional_call(data_rater, params, (input_ids, attention_mask, batch_sources))
    if isinstance(data_rater, MoEDataRater):
        return functional_call(data_rater, params, (input_ids, attention_mask))
    return functional_call(data_rater, params, (input_ids, attention_mask))


def _resolve_requested_sources(requested: Sequence[str], available: Sequence[str]) -> Tuple[List[str], List[str]]:
    norm_to_actual = {_normalize_source_name(src): str(src) for src in available}
    resolved = []
    missing = []
    for source_name in requested:
        actual = norm_to_actual.get(_normalize_source_name(source_name))
        if actual is None:
            missing.append(str(source_name))
            continue
        if actual not in resolved:
            resolved.append(actual)
    return resolved, missing


def _build_outer_sampler(
    val_loader,
    val_raw,
    outer_sampling: str,
    outer_per_source: Optional[int],
    hard_outer_sources: Optional[Sequence[str]],
):
    if outer_sampling == "random":
        return None

    dataset = getattr(val_loader, "dataset", None)
    collate_fn = getattr(val_loader, "collate_fn", None)
    if dataset is None or collate_fn is None:
        raise ValueError("outer_sampling requires val_loader to expose dataset and collate_fn.")
    if "raw_index" not in getattr(dataset, "column_names", []):
        raise ValueError("outer_sampling requires val dataset batches to include raw_index.")
    if val_raw is None or "source" not in getattr(val_raw, "column_names", []):
        raise ValueError("outer_sampling requires val_raw with source labels.")

    indices_by_source: Dict[str, List[int]] = {}
    for tokenized_idx, raw_idx_value in enumerate(dataset["raw_index"]):
        raw_idx = int(raw_idx_value.item()) if torch.is_tensor(raw_idx_value) else int(raw_idx_value)
        source_name = _source_from_raw_index(raw_idx, val_raw)
        if source_name == "__unknown__":
            continue
        indices_by_source.setdefault(source_name, []).append(int(tokenized_idx))

    available_sources = sorted(indices_by_source)
    if not available_sources:
        raise ValueError("No source-labeled validation samples available for outer_sampling.")

    if outer_sampling == "balanced":
        selected_sources = available_sources
        missing_sources = []
    elif outer_sampling == "harder":
        requested_sources = list(hard_outer_sources) if hard_outer_sources else list(DEFAULT_HARD_OUTER_SOURCES)
        selected_sources, missing_sources = _resolve_requested_sources(requested_sources, available_sources)
        if not selected_sources:
            raise ValueError(
                f"outer_sampling='harder' found no matching sources. Requested={requested_sources}, "
                f"available={available_sources}"
            )
    else:
        raise ValueError(f"Unsupported outer_sampling mode: {outer_sampling}")

    base_batch_size = int(getattr(val_loader, "batch_size", 1) or 1)
    per_source = int(outer_per_source) if outer_per_source is not None else max(1, base_batch_size // max(1, len(selected_sources)))
    if per_source <= 0:
        raise ValueError("outer_per_source must be positive when provided.")

    if missing_sources:
        logger.warning(
            "[meta] outer_sampling=%s missing requested sources: %s",
            outer_sampling,
            ", ".join(missing_sources),
        )
    logger.info(
        "[meta] outer_sampling=%s | selected_sources=%s | outer_per_source=%d | outer_batch_size=%d",
        outer_sampling,
        selected_sources,
        per_source,
        per_source * len(selected_sources),
    )

    return {
        "dataset": dataset,
        "collate_fn": collate_fn,
        "indices_by_source": indices_by_source,
        "selected_sources": selected_sources,
        "per_source": per_source,
        "mode": outer_sampling,
    }


def _sample_outer_batch(outer_sampler):
    if outer_sampler is None:
        raise ValueError("outer_sampler must be initialized before sampling.")

    sampled_indices: List[int] = []
    per_source = int(outer_sampler["per_source"])
    for source_name in outer_sampler["selected_sources"]:
        candidates = outer_sampler["indices_by_source"][source_name]
        if not candidates:
            continue
        if per_source <= len(candidates):
            chosen = random.sample(candidates, per_source)
        else:
            chosen = random.choices(candidates, k=per_source)
        sampled_indices.extend(int(idx) for idx in chosen)

    if not sampled_indices:
        raise RuntimeError("outer_sampling produced an empty outer batch.")

    random.shuffle(sampled_indices)
    samples = [outer_sampler["dataset"][idx] for idx in sampled_indices]
    return outer_sampler["collate_fn"](samples)


def _next_batch(batch_iter, loader):
    try:
        batch = next(batch_iter)
    except StopIteration:
        batch_iter = iter(loader)
        batch = next(batch_iter)
    return batch, batch_iter


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


def _zscore_unnormalize_preds(
    preds: torch.Tensor,
    raw_indices: torch.Tensor,
    raw_dataset,
    src2stats: Dict[str, Tuple[float, float]],
    global_mean: float,
    global_std: float,
) -> torch.Tensor:
    device = preds.device
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
    return preds * stds_t + means_t


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


def _first_order_weighted_grads(
    per_sample_loss: torch.Tensor,
    weights: torch.Tensor,
    params: Sequence[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], ...]:
    if per_sample_loss.ndim != 1:
        raise ValueError("per_sample_loss must be a 1D tensor.")
    if weights.shape != per_sample_loss.shape:
        raise ValueError("weights and per_sample_loss must have the same shape.")

    accum: List[Optional[torch.Tensor]] = [None] * len(params)
    batch_size = int(per_sample_loss.numel())

    for sample_idx in range(batch_size):
        sample_grads = torch.autograd.grad(
            per_sample_loss[sample_idx],
            tuple(params),
            retain_graph=sample_idx + 1 < batch_size,
            create_graph=False,
            allow_unused=True,
        )
        sample_weight = weights[sample_idx]
        for param_idx, sample_grad in enumerate(sample_grads):
            if sample_grad is None:
                continue
            contrib = sample_weight * sample_grad.detach()
            if accum[param_idx] is None:
                accum[param_idx] = contrib
            else:
                accum[param_idx] = accum[param_idx] + contrib

    return tuple(accum)


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
    if objective == "mse":
        return F.mse_loss(preds, targets)
    if objective == "rmse":
        return torch.sqrt(F.mse_loss(preds, targets) + outer_eps)
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
) -> nn.Module:
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
        allowed_objectives = {"mse", "rmse", "mse_norm", "pearson", "cosine", "mix", "source_stratified_mse"}
        if outer_objective not in allowed_objectives:
            raise ValueError(f"outer_objective must be one of {sorted(allowed_objectives)}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        if datarater_arch not in {"single", "multihead", "moe"}:
            raise ValueError("datarater_arch must be one of {'single', 'multihead', 'moe'}.")
        if outer_sampling not in {"random", "balanced", "harder"}:
            raise ValueError("outer_sampling must be one of {'random', 'balanced', 'harder'}.")
        if router_aux_loss_coef < 0.0:
            raise ValueError("router_aux_loss_coef must be non-negative.")
        if router_z_loss_coef < 0.0:
            raise ValueError("router_z_loss_coef must be non-negative.")
        if datarater_arch == "moe" and router_noise_std > 0.0:
            logger.warning(
                "[meta] p4-mixflow forces router_noise_std from %.4f to 0.0 so MixFlow recomputation stays deterministic.",
                router_noise_std,
            )
            router_noise_std = 0.0

        # v2: clamp T_backprop
        T_backprop_eff = min(T_backprop, T_window)
        T_warmup = T_window - T_backprop_eff  # Warmup steps are detached (no graph construction)
        logger.info(
            "[meta-v2] T_window=%d, T_backprop=%d, T_warmup=%d, datarater_arch=%s, outer_sampling=%s",
            T_window,
            T_backprop_eff,
            T_warmup,
            datarater_arch,
            outer_sampling,
        )

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

        source_names = infer_source_names(train_raw, val_raw)
        if datarater_arch == "multihead":
            logger.info("[meta] multi-head sources=%s", source_names)
            if len(source_names) <= 1:
                logger.warning(
                    "[meta] datarater_arch='multihead' but only %d source was found. "
                    "This run will behave like a single shared head; use data_mode='all' for real per-source heads.",
                    len(source_names),
                )
        if datarater_arch == "moe":
            logger.info(
                "[meta] moe config | experts=%d | top_k=%d | capacity_factor=%.3f | "
                "router_tau=%.3f | router_noise_std=%.3f | merge=%s | drop_overflow=%s | "
                "aux_coef=%.4f | z_coef=%.4f",
                num_experts,
                router_top_k,
                capacity_factor,
                router_temperature,
                router_noise_std,
                moe_score_merge,
                drop_overflow_tokens,
                router_aux_loss_coef,
                router_z_loss_coef,
            )

        # DataRater (heavy)
        data_rater = build_datarater_model(
            arch=datarater_arch,
            source_names=source_names,
            cache_init_state=True,
            force_eager_attn=force_eager_attn,
            num_experts=num_experts,
            router_top_k=router_top_k,
            capacity_factor=capacity_factor,
            router_temperature=router_temperature,
            router_noise_std=router_noise_std,
            moe_score_merge=moe_score_merge,
            drop_overflow_tokens=drop_overflow_tokens,
        ).to(device)
        rater_opt = torch.optim.Adam(data_rater.parameters(), lr=1e-4)

        # Inner population (heavy)
        population = [
            ESMForAffinity(cache_init_state=True, force_eager_attn=force_eager_attn).to(device)
            for _ in range(n_inner_models)
        ]

        train_iter = iter(train_loader)
        val_iter = iter(val_loader)
        outer_sampler = _build_outer_sampler(
            val_loader=val_loader,
            val_raw=val_raw,
            outer_sampling=outer_sampling,
            outer_per_source=outer_per_source,
            hard_outer_sources=hard_outer_sources,
        )

        if use_first_order_ablation:
            logger.warning(
                "[meta] p4-mixflow ignores --ablation; MixFlow-MG already uses exact local JVP backward without the old surrogate path."
            )

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
            outer_loss_values = []
            step_router_infos = []

            inner_batches = []
            for _ in range(T_window):
                x_in, train_iter = _next_batch(train_iter, train_loader)
                inner_batches.append(x_in)

            if outer_sampling == "random":
                x_out, val_iter = _next_batch(val_iter, val_loader)
            else:
                x_out = _sample_outer_batch(outer_sampler)

            if datarater_arch == "moe":
                for x_in in inner_batches:
                    input_ids = x_in["input_ids"].to(device)
                    mask = x_in["attention_mask"].to(device)
                    raw_index = x_in.get("raw_index")
                    if raw_index is not None:
                        raw_index = raw_index.to(device)
                    _, router_info = datarater_forward(
                        data_rater,
                        input_ids,
                        mask,
                        raw_indices=raw_index,
                        raw_dataset=train_raw,
                        return_router_info=True,
                    )
                    if router_info is not None:
                        step_router_infos.append(router_info)

            out_ids = x_out["input_ids"].to(device)
            out_mask = x_out["attention_mask"].to(device)
            out_targets = x_out["affinity"].to(device)
            out_raw_index = x_out.get("raw_index")
            if out_raw_index is not None:
                out_raw_index = out_raw_index.to(device)

            batch_src_std = None
            if outer_objective in {"mse_norm", "mix"}:
                if out_raw_index is not None and val_raw is not None:
                    batch_src_std = _lookup_batch_src_std(out_raw_index, val_raw, src2std, global_std, device)
                else:
                    batch_src_std = torch.full_like(out_targets, fill_value=float(global_std))

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

            params = tuple(data_rater.parameters())

            for m_idx in models_to_process:
                inner_model = population[m_idx]
                fast_weights = dict(inner_model.named_parameters())

                for inner_step_idx, x_in in enumerate(inner_batches):
                    input_ids = x_in["input_ids"].to(device)
                    mask = x_in["attention_mask"].to(device)
                    targets = x_in["affinity"].to(device)
                    raw_index = x_in.get("raw_index")
                    if raw_index is not None:
                        raw_index = raw_index.to(device)

                    if use_zscore_inner and src2zscore and raw_index is not None:
                        targets = _zscore_normalize_targets(
                            targets,
                            raw_index,
                            train_raw,
                            src2zscore,
                            zscore_global_mean,
                            zscore_global_std,
                        )

                    if inner_step_idx < T_warmup:
                        raw_scores = datarater_forward(
                            data_rater,
                            input_ids,
                            mask,
                            raw_indices=raw_index,
                            raw_dataset=train_raw,
                        )
                        weights = F.softmax(raw_scores / tau, dim=0).detach()
                        preds = functional_forward(inner_model, fast_weights, input_ids, mask)
                        per = F.mse_loss(preds.float(), targets.float(), reduction="none")
                        inner_loss = torch.sum(weights.float() * per)
                        grads = torch.autograd.grad(
                            inner_loss,
                            tuple(fast_weights.values()),
                            create_graph=False,
                            allow_unused=True,
                        )
                        grads = [g.detach() if g is not None else None for g in grads]
                        fast_weights = {
                            name: ((weight - inner_lr * grad_value) if grad_value is not None else weight).detach()
                            for (name, weight), grad_value in zip(fast_weights.items(), grads)
                        }
                    else:
                        fast_weights = mixflow_inner_update(
                            inner_model=inner_model,
                            data_rater=data_rater,
                            fast_weights=fast_weights,
                            input_ids=input_ids,
                            attention_mask=mask,
                            targets=targets,
                            tau=tau,
                            inner_lr=inner_lr,
                            functional_forward_fn=functional_forward,
                            functional_datarater_forward_fn=functional_datarater_forward,
                            raw_indices=raw_index,
                            raw_dataset=train_raw,
                        )

                outer_preds = functional_forward(inner_model, fast_weights, out_ids, out_mask)
                if use_zscore_inner and src2zscore and out_raw_index is not None and val_raw is not None:
                    outer_preds = _zscore_unnormalize_preds(
                        outer_preds,
                        out_raw_index,
                        val_raw,
                        src2zscore,
                        zscore_global_mean,
                        zscore_global_std,
                    )

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
                outer_loss_values.append(float(outer_loss.detach().item()))

                meta_grads = torch.autograd.grad(
                    outer_loss,
                    params,
                    allow_unused=True,
                )
                meta_grads_accumulator.append(meta_grads)

                with torch.no_grad():
                    for name, param in inner_model.named_parameters():
                        param.copy_(fast_weights[name].detach())

            for param, grads_for_param in zip(params, zip(*meta_grads_accumulator)):
                param.grad = _safe_mean(list(grads_for_param), param)

            router_reg = None
            step_router_info = None
            if datarater_arch == "moe":
                step_router_info = _aggregate_router_infos(step_router_infos, num_experts, device)
                if step_router_info is not None:
                    reg_terms = []
                    if router_aux_loss_coef > 0.0:
                        reg_terms.append(router_aux_loss_coef * step_router_info["aux_loss"])
                    if router_z_loss_coef > 0.0:
                        reg_terms.append(router_z_loss_coef * step_router_info["z_loss"])
                    if reg_terms:
                        router_reg = torch.stack(reg_terms).sum()

            if router_reg is not None:
                router_grads = torch.autograd.grad(
                    router_reg,
                    params,
                    allow_unused=True,
                )
                for param, grad_value in zip(params, router_grads):
                    if grad_value is None:
                        continue
                    if param.grad is None:
                        param.grad = grad_value
                    else:
                        param.grad = param.grad + grad_value

            rater_opt.step()

            if (step + 1) % 50 == 0:
                with torch.no_grad():
                    gsum = 0.0
                    for param in data_rater.parameters():
                        gsum += float(param.grad.norm().item()) if param.grad is not None else 0.0
                    outer_loss_value = (
                        sum(outer_loss_values) / float(len(outer_loss_values))
                        if outer_loss_values else 0.0
                    )
                    msg = (
                        f"[meta] step {step+1}/{n_meta_steps} | obj={outer_objective} "
                        f"| outer_loss={outer_loss_value:.6f} | grad_norm_sum={gsum:.4f}"
                    )
                    if batch_src_std is not None:
                        msg += f" | mean_src_std={float(batch_src_std.mean().item()):.6f}"
                    if datarater_arch == "moe" and step_router_info is not None:
                        expert_counts = step_router_info["expert_counts"].detach().cpu().tolist()
                        msg += (
                            f" | router_aux={float(step_router_info['aux_loss'].detach().item()):.6f}"
                            f" | router_z={float(step_router_info['z_loss'].detach().item()):.6f}"
                            f" | router_entropy={float(step_router_info['router_entropy'].detach().item()):.6f}"
                            f" | max_share={float(step_router_info['max_expert_share'].detach().item()):.4f}"
                            f" | overflow={int(step_router_info['overflow_count'].detach().item())}"
                            f" | dropped={int(step_router_info['dropped_tokens'].detach().item())}"
                            f" | capacity={int(step_router_info['capacity'].detach().item())}"
                            f" | expert_counts={expert_counts}"
                        )
                logger.info(msg)

        return data_rater

    finally:
        # always restore SDPA backend state
        sdp_ctx.__exit__(None, None, None)


# =========================
# 3) Filtering (non-differentiable by design)
# =========================
def filter_dataset(
    data_rater,
    original_dataset,
    raw_dataset=None,
    N_ref=10000,
    B=256,
    keep_ratio=0.7,
    return_indices: bool = False,
):
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
            raw_idx_value = sample.get("raw_index") if isinstance(sample, dict) else None
            raw_idx = None
            if raw_idx_value is not None:
                raw_idx_int = int(raw_idx_value.item()) if torch.is_tensor(raw_idx_value) else int(raw_idx_value)
                raw_idx = torch.tensor([raw_idx_int], dtype=torch.long, device=device)
            score = datarater_forward(
                data_rater,
                input_ids,
                mask,
                raw_indices=raw_idx,
                raw_dataset=raw_dataset,
            ).item()
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
            raw_idx_value = sample.get("raw_index") if isinstance(sample, dict) else None
            raw_idx = None
            if raw_idx_value is not None:
                raw_idx_int = int(raw_idx_value.item()) if torch.is_tensor(raw_idx_value) else int(raw_idx_value)
                raw_idx = torch.tensor([raw_idx_int], dtype=torch.long, device=device)

            score = datarater_forward(
                data_rater,
                input_ids,
                mask,
                raw_indices=raw_idx,
                raw_dataset=raw_dataset,
            ).item()
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
