from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from veomni.lora.moe_layers import LoraIndependentExperts, LoraSharedExperts
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type
from veomni.utils.import_utils import is_quack_gemm_available


class _TinyGptOssExperts(nn.Module):
    def __init__(self, *, experts: int = 4, hidden: int = 16, intermediate: int = 16):
        super().__init__()
        self.num_experts = experts
        self.hidden_size = hidden
        self.intermediate_size = intermediate
        self.gate_up_proj = nn.Parameter(torch.randn(experts, hidden, 2 * intermediate) * 0.05)
        self.gate_up_proj_bias = nn.Parameter(torch.randn(experts, 2 * intermediate) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(experts, intermediate, hidden) * 0.05)
        self.down_proj_bias = nn.Parameter(torch.randn(experts, hidden) * 0.02)
        self.alpha = 1.702
        self.limit = 7.0


def _reference(
    hidden_states,
    selected_experts,
    routing_weights,
    gate_up_proj,
    gate_up_proj_bias,
    down_proj,
    down_proj_bias,
    lora,
    scale,
    alpha=1.702,
    limit=7.0,
):
    num_experts = gate_up_proj.shape[0]
    output = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    gate_delta = F.linear(F.linear(hidden_states, lora[0]), lora[1]) * scale
    up_delta = F.linear(F.linear(hidden_states, lora[2]), lora[3]) * scale
    for expert_idx in range(num_experts):
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        if token_idx.numel() == 0:
            continue
        gate_up = hidden_states[token_idx] @ gate_up_proj[expert_idx] + gate_up_proj_bias[expert_idx]
        gate_up = gate_up.clone()
        gate_up[..., ::2] += gate_delta[token_idx]
        gate_up[..., 1::2] += up_delta[token_idx]
        gate = gate_up[..., ::2].clamp(max=limit)
        up = gate_up[..., 1::2].clamp(min=-limit, max=limit)
        mid = (up + 1) * (gate * torch.sigmoid(gate * alpha))
        expert_output = mid @ down_proj[expert_idx] + down_proj_bias[expert_idx]
        expert_output = expert_output + F.linear(F.linear(mid, lora[4]), lora[5]) * scale
        output.index_add_(
            0,
            token_idx,
            expert_output * routing_weights[token_idx, top_k_pos, None],
        )
    return output


def _l2_relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)).item()


def test_gpt_oss_shared_wrapper_preserves_interleaved_base_with_zero_adapter(monkeypatch):
    from veomni.lora import ops as lora_ops

    monkeypatch.setattr(lora_ops, "_gpt_oss_quack_lora_moe_forward", None)
    torch.manual_seed(0)
    base = _TinyGptOssExperts()
    wrapped = LoraSharedExperts(copy.deepcopy(base), r=4, lora_alpha=8)
    hidden = torch.randn(12, base.hidden_size)
    selected = torch.randint(0, base.num_experts, (12, 2))
    routing = torch.softmax(torch.randn(12, 2), dim=-1)

    zeros = tuple(wrapped.get_lora_A_weight(name) for name in ("gate_proj", "up_proj", "down_proj"))
    zeros_b = tuple(wrapped.get_lora_B_weight(name) for name in ("gate_proj", "up_proj", "down_proj"))
    lora = (zeros[0], zeros_b[0], zeros[1], zeros_b[1], zeros[2], zeros_b[2])
    expected = _reference(
        hidden,
        selected,
        routing,
        base.gate_up_proj,
        base.gate_up_proj_bias,
        base.down_proj,
        base.down_proj_bias,
        lora,
        wrapped._lora_scale_value,
    )
    actual = wrapped(hidden, selected, routing)

    torch.testing.assert_close(actual, expected)
    names = dict(wrapped.named_parameters())
    assert "gate_up_proj_bias" in names
    assert "down_proj_bias" in names


def test_gpt_oss_independent_expert_lora_is_rejected():
    with pytest.raises(NotImplementedError, match="share_expert_lora: true"):
        LoraIndependentExperts(_TinyGptOssExperts(), r=4, lora_alpha=8)


def test_gpt_oss_quack_shared_lora_matches_eager_forward_and_backward():
    if not IS_CUDA_AVAILABLE or not is_quack_gemm_available():
        pytest.skip("GPT-OSS Quack LoRA parity requires an SM90+ CUDA GPU and Quack.")

    from veomni.lora.ops.gpt_oss_quack import quack_gemm_gpt_oss_fused_lora_moe_forward

    torch.manual_seed(7)
    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    tokens, experts, hidden, intermediate, rank, top_k = 64, 4, 64, 64, 8, 2
    gate_up = torch.randn(experts, hidden, 2 * intermediate, device=device, dtype=dtype) * 0.03
    gate_up_bias = torch.randn(experts, 2 * intermediate, device=device, dtype=dtype) * 0.02
    down = torch.randn(experts, intermediate, hidden, device=device, dtype=dtype) * 0.03
    down_bias = torch.randn(experts, hidden, device=device, dtype=dtype) * 0.02
    selected = torch.randint(0, experts, (tokens, top_k), device=device)
    routing = torch.softmax(torch.randn(tokens, top_k, device=device), dim=-1).to(dtype)
    upstream = torch.randn(tokens, hidden, device=device, dtype=dtype)

    def run(*, fused: bool):
        torch.manual_seed(11)
        h = torch.randn(tokens, hidden, device=device, dtype=dtype).requires_grad_()
        lora = (
            (torch.randn(rank, hidden, device=device, dtype=dtype) * 0.02).requires_grad_(),
            (torch.randn(intermediate, rank, device=device, dtype=dtype) * 0.02).requires_grad_(),
            (torch.randn(rank, hidden, device=device, dtype=dtype) * 0.02).requires_grad_(),
            (torch.randn(intermediate, rank, device=device, dtype=dtype) * 0.02).requires_grad_(),
            (torch.randn(rank, intermediate, device=device, dtype=dtype) * 0.02).requires_grad_(),
            (torch.randn(hidden, rank, device=device, dtype=dtype) * 0.02).requires_grad_(),
        )
        if fused:
            output = quack_gemm_gpt_oss_fused_lora_moe_forward(
                experts,
                routing,
                selected,
                h,
                gate_up,
                gate_up_bias,
                down,
                down_bias,
                *lora,
                0.5,
                0.5,
                0.5,
            )
        else:
            output = _reference(
                h,
                selected,
                routing,
                gate_up,
                gate_up_bias,
                down,
                down_bias,
                lora,
                0.5,
            )
        grads = torch.autograd.grad(output, (h, *lora), grad_outputs=upstream)
        return output.detach(), tuple(grad.detach() for grad in grads)

    expected_output, expected_grads = run(fused=False)
    actual_output, actual_grads = run(fused=True)
    assert _l2_relative(actual_output, expected_output) < 0.025
    for actual, expected in zip(actual_grads, expected_grads):
        assert _l2_relative(actual, expected) < 0.04
