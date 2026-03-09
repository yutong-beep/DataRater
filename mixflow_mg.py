import torch
import torch.nn.functional as F
from torch.func import grad, jvp


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


def _split_datarater_output(output):
    if isinstance(output, tuple):
        if len(output) == 2:
            return output[0], output[1]
        raise ValueError("Unexpected datarater output tuple length.")
    return output, None


def compute_inner_weights(raw_scores, tau, router_info=None):
    if router_info is not None and router_info.get("expert_scores") is not None:
        expert_scores = router_info["expert_scores"]
        batch_size = int(expert_scores.size(0))
        final_weights = torch.zeros(batch_size, dtype=expert_scores.dtype, device=expert_scores.device)
        for expert_idx in range(int(expert_scores.size(1))):
            expert_column = expert_scores[:, expert_idx]
            valid_mask = torch.isfinite(expert_column)
            valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(-1)
            if valid_indices.numel() == 0:
                continue
            expert_weights = F.softmax(expert_column.index_select(0, valid_indices) / tau, dim=0)
            final_weights = final_weights.index_add(0, valid_indices, expert_weights)

        total_weight = final_weights.sum()
        fallback = torch.full_like(final_weights, 1.0 / max(batch_size, 1))
        return torch.where(
            total_weight > 0,
            final_weights / total_weight.clamp(min=1e-9),
            fallback,
        )

    return F.softmax(raw_scores / tau, dim=0)


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
            raw_output = functional_datarater_forward_fn(
                data_rater,
                rater_dict,
                input_ids,
                attention_mask,
                raw_indices=ctx.raw_indices,
                raw_dataset=raw_dataset,
                return_router_info=True,
            )
            raw_scores, router_info = _split_datarater_output(raw_output)
            weights = compute_inner_weights(raw_scores, tau, router_info=router_info)

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

            raw_output = ctx.functional_datarater_forward_fn(
                ctx.data_rater,
                rater_dict,
                input_ids,
                attention_mask,
                raw_indices=raw_indices,
                raw_dataset=ctx.raw_dataset,
                return_router_info=True,
            )
            raw_scores, router_info = _split_datarater_output(raw_output)
            weights = compute_inner_weights(raw_scores, ctx.tau, router_info=router_info)

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

        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            *grad_fw,
            *grad_rater,
        )


def mixflow_inner_update(
    inner_model,
    data_rater,
    fast_weights,
    input_ids,
    attention_mask,
    targets,
    tau,
    inner_lr,
    functional_forward_fn,
    functional_datarater_forward_fn,
    raw_indices=None,
    raw_dataset=None,
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
        *fw_values,
        *rater_values,
    )

    return dict(zip(fast_weight_keys, new_fw_values))
