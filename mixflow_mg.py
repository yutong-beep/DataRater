import torch
import torch.nn.functional as F
from torch.func import grad, jvp

from datarater_weighting import (
    apply_source_score_bias,
    apply_source_weight_cap,
    compute_inner_weights,
)


def _replace_none_with_zeros(values, refs):
    out = []
    for value, ref in zip(values, refs):
        if value is None:
            out.append(torch.zeros_like(ref))
        else:
            out.append(value)
    return tuple(out)


def _build_param_dict(keys, values):
    return {key: value for key, value in zip(keys, values)}


class MixFlowMGInnerStep(torch.autograd.Function):
    """
    One inner SGD step with exact local Jacobian-vector products.
    """

    @staticmethod
    def forward(
        ctx,
        inner_lr,
        tau,
        fast_weight_keys,
        rater_keys,
        inner_model,
        data_rater,
        functional_forward_fn,
        functional_datarater_forward_fn,
        raw_indices,
        raw_dataset,
        input_ids,
        attention_mask,
        targets,
        precomputed_raw_scores,
        precomputed_source_labels,
        weighting_mode,
        inner_source_score_bias,
        inner_source_weight_cap,
        score_within_source_std_floor,
        score_within_source_std_penalty_coef,
        score_source_bias_penalty_coef,
        *all_values,
    ):
        num_fw = len(fast_weight_keys)
        fw_values = all_values[:num_fw]
        rater_values = all_values[num_fw:]

        ctx.inner_lr = inner_lr
        ctx.tau = tau
        ctx.fast_weight_keys = tuple(fast_weight_keys)
        ctx.rater_keys = tuple(rater_keys)
        ctx.inner_model = inner_model
        ctx.data_rater = data_rater
        ctx.functional_forward_fn = functional_forward_fn
        ctx.functional_datarater_forward_fn = functional_datarater_forward_fn
        ctx.num_fw = num_fw
        ctx.raw_indices = raw_indices.detach() if torch.is_tensor(raw_indices) else None
        ctx.raw_dataset = raw_dataset
        ctx.precomputed_source_labels = list(precomputed_source_labels) if precomputed_source_labels is not None else None
        ctx.weighting_mode = weighting_mode
        ctx.inner_source_score_bias = dict(inner_source_score_bias) if inner_source_score_bias is not None else None
        ctx.inner_source_weight_cap = dict(inner_source_weight_cap) if inner_source_weight_cap is not None else None
        ctx.score_within_source_std_floor = float(score_within_source_std_floor)
        ctx.score_within_source_std_penalty_coef = float(score_within_source_std_penalty_coef)
        ctx.score_source_bias_penalty_coef = float(score_source_bias_penalty_coef)

        ctx.save_for_backward(
            input_ids,
            attention_mask,
            targets,
            *fw_values,
            *rater_values,
        )

        fw_req = tuple(weight.detach().requires_grad_(True) for weight in fw_values)
        rater_req = tuple(param.detach().requires_grad_(True) for param in rater_values)

        fw_dict = _build_param_dict(fast_weight_keys, fw_req)
        rater_dict = _build_param_dict(rater_keys, rater_req)

        with torch.enable_grad():
            if precomputed_raw_scores is None:
                raw_scores = functional_datarater_forward_fn(
                    data_rater,
                    rater_dict,
                    input_ids,
                    attention_mask,
                    raw_indices=ctx.raw_indices,
                    raw_dataset=raw_dataset,
                )
            else:
                raw_scores = precomputed_raw_scores.detach()
            source_labels = ctx.precomputed_source_labels or []
            raw_scores = apply_source_score_bias(raw_scores, source_labels, ctx.inner_source_score_bias)
            weights = compute_inner_weights(raw_scores, tau=tau, weighting_mode=weighting_mode)
            weights, _ = apply_source_weight_cap(weights, source_labels, ctx.inner_source_weight_cap)
            preds = functional_forward_fn(inner_model, fw_dict, input_ids, attention_mask)
            per_sample_loss = F.mse_loss(preds.float(), targets.float(), reduction="none")
            inner_loss = torch.sum(weights.float() * per_sample_loss)

            grads = torch.autograd.grad(
                inner_loss,
                fw_req,
                create_graph=False,
                allow_unused=True,
            )

        grads = _replace_none_with_zeros(grads, fw_values)
        return tuple(weight - inner_lr * grad_value for weight, grad_value in zip(fw_values, grads))

    @staticmethod
    def backward(ctx, *grad_outputs):
        saved = ctx.saved_tensors
        input_ids = saved[0]
        attention_mask = saved[1]
        targets = saved[2]

        num_fw = ctx.num_fw
        fw_values = tuple(saved[3 : 3 + num_fw])
        rater_values = tuple(saved[3 + num_fw :])
        raw_indices = ctx.raw_indices

        v = _replace_none_with_zeros(grad_outputs, fw_values)

        fw_base = tuple(weight.detach().requires_grad_(True) for weight in fw_values)
        rater_base = tuple(param.detach().requires_grad_(True) for param in rater_values)

        def compute_inner_loss(fw_tuple, r_tuple):
            fw_dict = _build_param_dict(ctx.fast_weight_keys, fw_tuple)
            rater_dict = _build_param_dict(ctx.rater_keys, r_tuple)

            raw_scores = ctx.functional_datarater_forward_fn(
                ctx.data_rater,
                rater_dict,
                input_ids,
                attention_mask,
                raw_indices=raw_indices,
                raw_dataset=ctx.raw_dataset,
            )
            source_labels = ctx.precomputed_source_labels or []
            raw_scores = apply_source_score_bias(raw_scores, source_labels, ctx.inner_source_score_bias)
            weights = compute_inner_weights(raw_scores, tau=ctx.tau, weighting_mode=ctx.weighting_mode)
            weights, _ = apply_source_weight_cap(weights, source_labels, ctx.inner_source_weight_cap)
            preds = ctx.functional_forward_fn(ctx.inner_model, fw_dict, input_ids, attention_mask)
            per_sample_loss = F.mse_loss(preds.float(), targets.float(), reduction="none")
            return torch.sum(weights.float() * per_sample_loss)

        def grad_theta_fn(fw_tuple):
            return grad(compute_inner_loss, argnums=0)(fw_tuple, rater_base)

        def grad_eta_fn(fw_tuple):
            return grad(compute_inner_loss, argnums=1)(fw_tuple, rater_base)

        _, h_theta_v = jvp(
            grad_theta_fn,
            (fw_base,),
            (v,),
        )
        _, h_eta_v = jvp(
            grad_eta_fn,
            (fw_base,),
            (v,),
        )

        h_theta_v = _replace_none_with_zeros(h_theta_v, fw_values)
        h_eta_v = _replace_none_with_zeros(h_eta_v, rater_values)

        grad_fw = tuple(v_i - ctx.inner_lr * h_i for v_i, h_i in zip(v, h_theta_v))
        grad_rater = tuple(-ctx.inner_lr * h_i for h_i in h_eta_v)

        return (None,) * 21 + grad_fw + grad_rater


def mixflow_inner_update(
    inner_model,
    data_rater,
    fast_weights,
    input_ids,
    attention_mask,
    targets,
    tau,
    weighting_mode,
    inner_lr,
    functional_forward_fn,
    functional_datarater_forward_fn,
    raw_indices=None,
    raw_dataset=None,
    precomputed_raw_scores=None,
    precomputed_source_labels=None,
    inner_source_score_bias=None,
    inner_source_weight_cap=None,
    score_within_source_std_floor: float = 0.0,
    score_within_source_std_penalty_coef: float = 0.0,
    score_source_bias_penalty_coef: float = 0.0,
):
    fast_weight_keys = tuple(fast_weights.keys())
    fw_values = tuple(fast_weights.values())

    rater_named = dict(data_rater.named_parameters())
    rater_keys = tuple(rater_named.keys())
    rater_values = tuple(rater_named.values())

    new_fw_values = MixFlowMGInnerStep.apply(
        inner_lr,
        tau,
        fast_weight_keys,
        rater_keys,
        inner_model,
        data_rater,
        functional_forward_fn,
        functional_datarater_forward_fn,
        raw_indices,
        raw_dataset,
        input_ids,
        attention_mask,
        targets,
        precomputed_raw_scores,
        precomputed_source_labels,
        weighting_mode,
        inner_source_score_bias,
        inner_source_weight_cap,
        score_within_source_std_floor,
        score_within_source_std_penalty_coef,
        score_source_bias_penalty_coef,
        *fw_values,
        *rater_values,
    )

    return dict(zip(fast_weight_keys, new_fw_values))
