# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quack GPT-OSS MoE forward with shared LoRA on interleaved experts."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from quack.gemm_interface import gemm

from ...distributed.moe import preprocess, token_pre_all2all, tokens_post_all2all
from ...distributed.parallel_state import get_parallel_state
from ...ops.kernels.moe._kernels.kernel.moe import moe_gather, moe_scatter
from ...ops.kernels.moe.quack_gemm import _build_moe_indices, _cumsum_to_cu_seqlens
from ...ops.kernels.moe.quack_gemm_interleave_gate_up import (
    _assert_contiguous,
    _gpt_oss_mlp_activation,
    _gpt_oss_mlp_activation_backward,
    _segment_sum,
)


def _interleave_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    output = torch.empty((*gate.shape[:-1], gate.shape[-1] * 2), dtype=gate.dtype, device=gate.device)
    output[..., ::2] = gate
    output[..., 1::2] = up
    return output


def _shared_lora_forward(
    inp: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    tmp = F.linear(inp, lora_a)
    return tmp, F.linear(tmp, lora_b) * scale


def _shared_lora_backward(
    grad_delta: torch.Tensor,
    inp: torch.Tensor,
    tmp: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_tmp = F.linear(grad_delta, lora_b.t()) * scale
    grad_lora_b = grad_delta.t().to(tmp.dtype) @ tmp * scale
    grad_lora_a = grad_tmp.t().to(inp.dtype) @ inp
    grad_inp = F.linear(grad_tmp, lora_a.t())
    return grad_lora_a, grad_lora_b, grad_inp


class GptOssQuackSharedLoraMoeFunction(torch.autograd.Function):
    """Non-EP GPT-OSS Quack MoE with one LoRA pair shared by all experts."""

    @staticmethod
    def forward(
        ctx,
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
        lora_a_gate,
        lora_b_gate,
        lora_a_up,
        lora_b_up,
        lora_a_down,
        lora_b_down,
        lora_scale_gate,
        lora_scale_up,
        lora_scale_down,
        alpha,
        limit,
    ):
        cu_seqlens_m, A_idx, scatter_index = _build_moe_indices(selected_experts, num_experts)
        scatter_output = moe_scatter(hidden_states, scatter_index)

        gate_up = gemm(
            hidden_states,
            gate_up_proj,
            bias=gate_up_proj_bias,
            cu_seqlens_m=cu_seqlens_m,
            A_idx=A_idx,
            tuned=False,
        )
        tmp_gate, delta_gate = _shared_lora_forward(scatter_output, lora_a_gate, lora_b_gate, lora_scale_gate)
        tmp_up, delta_up = _shared_lora_forward(scatter_output, lora_a_up, lora_b_up, lora_scale_up)
        gate_up = gate_up + _interleave_gate_up(delta_gate, delta_up)
        fc1_activation = _gpt_oss_mlp_activation(gate_up, alpha, limit)

        fc2_output = gemm(
            fc1_activation,
            down_proj,
            bias=down_proj_bias,
            cu_seqlens_m=cu_seqlens_m,
            tuned=False,
        )
        tmp_down, delta_down = _shared_lora_forward(fc1_activation, lora_a_down, lora_b_down, lora_scale_down)
        fc2_output = fc2_output + delta_down

        reshaped_gate_weight = routing_weights.to(hidden_states.dtype).reshape(-1, 1)
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight
        expert_output = moe_gather(fc2_output * scattered_gate_weight, scatter_index)

        ctx.alpha = alpha
        ctx.limit = limit
        ctx.lora_scale_gate = lora_scale_gate
        ctx.lora_scale_up = lora_scale_up
        ctx.lora_scale_down = lora_scale_down
        ctx.gate_up_bias_requires_grad = gate_up_proj_bias.requires_grad
        ctx.down_bias_requires_grad = down_proj_bias.requires_grad
        ctx.save_for_backward(
            routing_weights,
            hidden_states,
            gate_up_proj,
            down_proj,
            scatter_index,
            cu_seqlens_m,
            scatter_output,
            gate_up,
            fc1_activation,
            fc2_output,
            scattered_gate_weight,
            lora_a_gate,
            lora_b_gate,
            lora_a_up,
            lora_b_up,
            lora_a_down,
            lora_b_down,
            tmp_gate,
            tmp_up,
            tmp_down,
        )
        return expert_output.reshape(hidden_states.shape)

    @staticmethod
    def backward(ctx, grad_output):
        (
            routing_weights,
            hidden_states,
            gate_up_proj,
            down_proj,
            scatter_index,
            cu_seqlens_m,
            scatter_output,
            gate_up,
            fc1_activation,
            fc2_output,
            scattered_gate_weight,
            lora_a_gate,
            lora_b_gate,
            lora_a_up,
            lora_b_up,
            lora_a_down,
            lora_b_down,
            tmp_gate,
            tmp_up,
            tmp_down,
        ) = ctx.saved_tensors

        grad_output = grad_output.view(-1, grad_output.shape[-1])
        grad_fc2_weighted = moe_scatter(grad_output, scatter_index)
        grad_scattered_weight = torch.sum(fc2_output * grad_fc2_weighted, dim=-1)
        grad_routing_weights = grad_scattered_weight[scatter_index.flatten()].reshape(routing_weights.shape)
        grad_fc2_output = grad_fc2_weighted * scattered_gate_weight

        grad_lora_a_down, grad_lora_b_down, grad_fc1_lora = _shared_lora_backward(
            grad_fc2_output,
            fc1_activation,
            tmp_down,
            lora_a_down,
            lora_b_down,
            ctx.lora_scale_down,
        )
        _assert_contiguous(down_proj, "down_proj")
        grad_fc1_activation = gemm(
            grad_fc2_output,
            down_proj.transpose(1, 2),
            cu_seqlens_m=cu_seqlens_m,
            tuned=False,
        )
        grad_fc1_activation = grad_fc1_activation + grad_fc1_lora
        grad_gate_up = _gpt_oss_mlp_activation_backward(grad_fc1_activation, gate_up, ctx.alpha, ctx.limit)

        grad_gate = grad_gate_up[..., ::2].contiguous()
        grad_up = grad_gate_up[..., 1::2].contiguous()
        grad_lora_a_gate, grad_lora_b_gate, grad_input_gate = _shared_lora_backward(
            grad_gate,
            scatter_output,
            tmp_gate,
            lora_a_gate,
            lora_b_gate,
            ctx.lora_scale_gate,
        )
        grad_lora_a_up, grad_lora_b_up, grad_input_up = _shared_lora_backward(
            grad_up,
            scatter_output,
            tmp_up,
            lora_a_up,
            lora_b_up,
            ctx.lora_scale_up,
        )
        _assert_contiguous(gate_up_proj, "gate_up_proj")
        grad_scatter_output = gemm(
            grad_gate_up,
            gate_up_proj.transpose(1, 2),
            cu_seqlens_m=cu_seqlens_m,
            tuned=False,
        )
        grad_scatter_output = grad_scatter_output + grad_input_gate + grad_input_up
        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index).reshape(hidden_states.shape)

        return (
            None,
            grad_routing_weights,
            None,
            grad_hidden_states,
            None,
            _segment_sum(grad_gate_up, cu_seqlens_m) if ctx.gate_up_bias_requires_grad else None,
            None,
            _segment_sum(grad_fc2_output, cu_seqlens_m) if ctx.down_bias_requires_grad else None,
            grad_lora_a_gate,
            grad_lora_b_gate,
            grad_lora_a_up,
            grad_lora_b_up,
            grad_lora_a_down,
            grad_lora_b_down,
            None,
            None,
            None,
            None,
            None,
        )


class EPGptOssQuackSharedLoraGroupGemm(torch.autograd.Function):
    """EP-local GPT-OSS expert GEMMs with shared LoRA."""

    @staticmethod
    def forward(
        ctx,
        permute_tokens,
        cumsum,
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
        lora_a_gate,
        lora_b_gate,
        lora_a_up,
        lora_b_up,
        lora_a_down,
        lora_b_down,
        lora_scale_gate,
        lora_scale_up,
        lora_scale_down,
        alpha,
        limit,
    ):
        cu_seqlens_m = _cumsum_to_cu_seqlens(cumsum)
        gate_up = gemm(
            permute_tokens,
            gate_up_proj,
            bias=gate_up_proj_bias,
            cu_seqlens_m=cu_seqlens_m,
            tuned=False,
        )
        tmp_gate, delta_gate = _shared_lora_forward(permute_tokens, lora_a_gate, lora_b_gate, lora_scale_gate)
        tmp_up, delta_up = _shared_lora_forward(permute_tokens, lora_a_up, lora_b_up, lora_scale_up)
        gate_up = gate_up + _interleave_gate_up(delta_gate, delta_up)
        fc1_activation = _gpt_oss_mlp_activation(gate_up, alpha, limit)

        output = gemm(
            fc1_activation,
            down_proj,
            bias=down_proj_bias,
            cu_seqlens_m=cu_seqlens_m,
            tuned=False,
        )
        tmp_down, delta_down = _shared_lora_forward(fc1_activation, lora_a_down, lora_b_down, lora_scale_down)
        output = output + delta_down

        ctx.alpha = alpha
        ctx.limit = limit
        ctx.lora_scale_gate = lora_scale_gate
        ctx.lora_scale_up = lora_scale_up
        ctx.lora_scale_down = lora_scale_down
        ctx.gate_up_bias_requires_grad = gate_up_proj_bias.requires_grad
        ctx.down_bias_requires_grad = down_proj_bias.requires_grad
        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            gate_up_proj,
            down_proj,
            gate_up,
            fc1_activation,
            lora_a_gate,
            lora_b_gate,
            lora_a_up,
            lora_b_up,
            lora_a_down,
            lora_b_down,
            tmp_gate,
            tmp_up,
            tmp_down,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            permute_tokens,
            cumsum,
            gate_up_proj,
            down_proj,
            gate_up,
            fc1_activation,
            lora_a_gate,
            lora_b_gate,
            lora_a_up,
            lora_b_up,
            lora_a_down,
            lora_b_down,
            tmp_gate,
            tmp_up,
            tmp_down,
        ) = ctx.saved_tensors
        cu_seqlens_m = _cumsum_to_cu_seqlens(cumsum)

        grad_lora_a_down, grad_lora_b_down, grad_fc1_lora = _shared_lora_backward(
            grad_output,
            fc1_activation,
            tmp_down,
            lora_a_down,
            lora_b_down,
            ctx.lora_scale_down,
        )
        _assert_contiguous(down_proj, "down_proj")
        grad_fc1_activation = gemm(
            grad_output,
            down_proj.transpose(1, 2),
            cu_seqlens_m=cu_seqlens_m,
            tuned=False,
        )
        grad_fc1_activation = grad_fc1_activation + grad_fc1_lora
        grad_gate_up = _gpt_oss_mlp_activation_backward(grad_fc1_activation, gate_up, ctx.alpha, ctx.limit)

        grad_gate = grad_gate_up[..., ::2].contiguous()
        grad_up = grad_gate_up[..., 1::2].contiguous()
        grad_lora_a_gate, grad_lora_b_gate, grad_input_gate = _shared_lora_backward(
            grad_gate,
            permute_tokens,
            tmp_gate,
            lora_a_gate,
            lora_b_gate,
            ctx.lora_scale_gate,
        )
        grad_lora_a_up, grad_lora_b_up, grad_input_up = _shared_lora_backward(
            grad_up,
            permute_tokens,
            tmp_up,
            lora_a_up,
            lora_b_up,
            ctx.lora_scale_up,
        )
        _assert_contiguous(gate_up_proj, "gate_up_proj")
        grad_permute_tokens = gemm(
            grad_gate_up,
            gate_up_proj.transpose(1, 2),
            cu_seqlens_m=cu_seqlens_m,
            tuned=False,
        )
        grad_permute_tokens = grad_permute_tokens + grad_input_gate + grad_input_up

        return (
            grad_permute_tokens,
            None,
            None,
            _segment_sum(grad_gate_up, cu_seqlens_m) if ctx.gate_up_bias_requires_grad else None,
            None,
            _segment_sum(grad_output, cu_seqlens_m) if ctx.down_bias_requires_grad else None,
            grad_lora_a_gate,
            grad_lora_b_gate,
            grad_lora_a_up,
            grad_lora_b_up,
            grad_lora_a_down,
            grad_lora_b_down,
            None,
            None,
            None,
            None,
            None,
        )


def quack_gemm_gpt_oss_fused_lora_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
    lora_a_gate: torch.Tensor,
    lora_b_gate: torch.Tensor,
    lora_a_up: torch.Tensor,
    lora_b_up: torch.Tensor,
    lora_a_down: torch.Tensor,
    lora_b_down: torch.Tensor,
    lora_scale_gate: float,
    lora_scale_up: float,
    lora_scale_down: float,
    alpha: float = 1.702,
    limit: float = 7.0,
):
    """Run shared GPT-OSS MoE-LoRA through Quack, including EP all-to-all."""
    if get_parallel_state().ep_enabled:
        expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
        input_splits, output_splits, local_tokens, local_token_cumsum = preprocess(
            expert_mask=expert_mask,
            num_experts=num_experts,
            ep_group=get_parallel_state().ep_group,
        )
        permute_tokens, routing_map, local_mapping, original_shape = token_pre_all2all(
            hidden_states=hidden_states,
            expert_mask=expert_mask,
            num_experts=num_experts,
            input_splits=input_splits,
            output_splits=output_splits,
            num_global_tokens_per_local_expert=local_tokens,
            ep_group=get_parallel_state().ep_group,
        )
        cumsum = torch.cumsum(local_token_cumsum, dim=0).to(permute_tokens.device)
        expert_outputs = EPGptOssQuackSharedLoraGroupGemm.apply(
            permute_tokens,
            cumsum,
            gate_up_proj,
            gate_up_proj_bias,
            down_proj,
            down_proj_bias,
            lora_a_gate,
            lora_b_gate,
            lora_a_up,
            lora_b_up,
            lora_a_down,
            lora_b_down,
            lora_scale_gate,
            lora_scale_up,
            lora_scale_down,
            alpha,
            limit,
        )
        return tokens_post_all2all(
            expert_outputs=expert_outputs,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            num_experts=num_experts,
            input_splits=input_splits,
            output_splits=output_splits,
            num_global_tokens_per_local_expert=local_tokens,
            routing_map=routing_map,
            local_input_permutation_mapping=local_mapping,
            org_hidden_states_shape=original_shape,
            ep_group=get_parallel_state().ep_group,
        )

    return GptOssQuackSharedLoraMoeFunction.apply(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
        lora_a_gate,
        lora_b_gate,
        lora_a_up,
        lora_b_up,
        lora_a_down,
        lora_b_down,
        lora_scale_gate,
        lora_scale_up,
        lora_scale_down,
        alpha,
        limit,
    )
