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

"""Fused MoE-LoRA kernels and their dispatch, owned by the native LoRA stack.

There is one kernel implementation per supported hardware/backend layout, all owned here so only
the LoRA math lives in this package — never the low-level GEMM/dispatch:

  * ``triton`` (GPU) — :mod:`.moe_group_gemm`: four ``autograd.Function``
    classes (hand-written forward + backward) reusing the shared grouped-gemm /
    scatter / gather Triton primitives under ``veomni.ops.kernels.moe._kernels``
    (the same primitives the non-LoRA fused MoE kernel uses).
  * ``npu`` — :mod:`.npu_moe_group_gemm`: composes the base NPU MoE primitives
    (``dispatch_preprocess`` / ``alltoall_dispatch`` / ``alltoall_combine`` +
    the ``npu_group_gemm`` autograd ``Function``) from
    ``veomni.ops.kernels.moe.npu_group_gemm`` and injects the LoRA deltas;
    backward is derived by autograd (no hand-written kernel).
  * ``quack`` (GPT-OSS) — :mod:`.gpt_oss_quack`: hand-written forward and
    backward for GPT-OSS's interleaved gate/up layout, including EP dispatch.

Dispatch model
--------------
Whether a LoRA-aware fused kernel is available is a property of the active
*base* MoE backend (``triton`` / ``quack`` / ``npu``), so binding is driven by
:func:`veomni.ops.kernels.moe.apply_veomni_fused_moe_patch`: after it selects a
base backend it calls :func:`bind_lora_moe_kernels` here to point (or clear) the
module-level ``_fused_lora_moe_forward`` / ``_fused_independent_lora_moe_forward``
pointers. ``triton`` (GPU) and ``npu`` ship the standard LoRA kernels, while
``quack`` binds the GPT-OSS-specific shared-LoRA kernel. All three include an
EP-aware path. Other Quack model layouts still have no LoRA-aware kernel.

The MoE-LoRA wrappers read the pointers directly (``from . import ops`` inside
their ``forward``); the public :func:`fused_lora_moe_forward` /
:func:`fused_independent_lora_moe_forward` dispatchers add the dtype guard and a
clear error when no kernel is bound.
"""

from __future__ import annotations

import torch


# Function pointers for the LoRA-aware fused MoE paths. ``None`` means "no
# fused-LoRA kernel bound for the active ``moe_implementation``" — in that case
# the wrappers in ``veomni.lora.moe_layers`` keep using their eager forwards.
# The standard pointers are bound for ``triton`` (GPU) and ``npu``. Quack
# leaves those two clear and binds the GPT-OSS-specific pointer below.
#
# * ``_fused_lora_moe_forward`` — Mode 2 (shared LoRA across experts), used by
#   ``LoraSharedExperts``.
# * ``_fused_independent_lora_moe_forward`` — Mode 1 (independent per-expert
#   LoRA), used by ``LoraIndependentExperts``.
_fused_lora_moe_forward = None
_fused_independent_lora_moe_forward = None
# GPT-OSS uses an interleaved ``[E, H, 2I]`` gate/up layout and a different
# activation, so its Quack LoRA kernel cannot share the standard pointer above.
_gpt_oss_quack_lora_moe_forward = None


def bind_lora_moe_kernels(fused_moe_kernel: str) -> None:
    """Bind (or clear) the LoRA fused-MoE pointers to match the base MoE backend.

    Called by :func:`veomni.ops.kernels.moe.apply_veomni_fused_moe_patch` right
    after it selects the base ``_fused_moe_forward`` kernel. Triton and NPU
    bind the standard layouts; Quack binds the GPT-OSS shared-LoRA layout.

    The kernel import is local so this stays free of an ``ops`` <-> ``lora``
    import cycle (the base MoE patch calls this lazily).
    """
    global _fused_lora_moe_forward, _fused_independent_lora_moe_forward
    global _gpt_oss_quack_lora_moe_forward
    if fused_moe_kernel == "triton":
        from .moe_group_gemm import (
            group_gemm_fused_independent_lora_moe_forward,
            group_gemm_fused_lora_moe_forward,
        )

        _fused_lora_moe_forward = group_gemm_fused_lora_moe_forward
        _fused_independent_lora_moe_forward = group_gemm_fused_independent_lora_moe_forward
        _gpt_oss_quack_lora_moe_forward = None
    elif fused_moe_kernel == "npu":
        # NPU reuses its base fused-MoE all-to-all EP dispatch/combine and adds
        # the seed-style LoRA deltas on the dispatched tokens — MoE-LoRA and EP
        # are orthogonal, so no LoRA-specific communication is introduced. The
        # kernel lives alongside the Triton one under ``veomni.lora.ops``.
        from .npu_moe_group_gemm import (
            npu_fused_independent_lora_moe_forward,
            npu_fused_lora_moe_forward,
        )

        _fused_lora_moe_forward = npu_fused_lora_moe_forward
        _fused_independent_lora_moe_forward = npu_fused_independent_lora_moe_forward
        _gpt_oss_quack_lora_moe_forward = None
    elif fused_moe_kernel == "quack":
        from .gpt_oss_quack import quack_gemm_gpt_oss_fused_lora_moe_forward

        # The standard Qwen-style pointers stay clear: only the GPT-OSS
        # interleaved shared-LoRA wrapper may dispatch to this kernel.
        _fused_lora_moe_forward = None
        _fused_independent_lora_moe_forward = None
        _gpt_oss_quack_lora_moe_forward = quack_gemm_gpt_oss_fused_lora_moe_forward
    else:
        # Unknown/unsupported backend: leave every LoRA dispatcher unbound.
        _fused_lora_moe_forward = None
        _fused_independent_lora_moe_forward = None
        _gpt_oss_quack_lora_moe_forward = None


def fused_lora_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    lora_a_gate: torch.Tensor,
    lora_b_gate: torch.Tensor,
    lora_a_up: torch.Tensor,
    lora_b_up: torch.Tensor,
    lora_a_down: torch.Tensor,
    lora_b_down: torch.Tensor,
    lora_scale_gate: float,
    lora_scale_up: float,
    lora_scale_down: float,
):
    """Public dispatcher for the fused MoE forward + shared seed-style two-LoRA (Mode 2).

    Mirrors ``veomni.ops.kernels.moe.fused_moe_forward`` for the fused experts
    layout: a single ``fc1_1_2_weight`` (``[E, 2I, H]``) plus two independent
    rank-r LoRA pairs (``gate`` + ``up``) covering the gate_up halves, plus the
    down LoRA pair. See
    :func:`veomni.lora.ops.moe_group_gemm.group_gemm_fused_lora_moe_forward`.

    Raises ``NotImplementedError`` if no LoRA-aware fused kernel is bound for
    the active ``moe_implementation`` — callers (see
    :class:`veomni.lora.moe_layers.LoraSharedExperts`) are expected to check
    this module's ``_fused_lora_moe_forward`` is non-``None`` before calling.
    """
    if _fused_lora_moe_forward is None:
        raise NotImplementedError(
            "No fused MoE-LoRA kernel is bound. Set ops_implementation.moe_implementation "
            "to a backend that ships a LoRA variant ('fused_triton' on GPU, 'fused_npu' on NPU) "
            "or fall back to the eager LoRA forward."
        )

    assert routing_weights.dtype in [torch.bfloat16, torch.float16], (
        f"routing_weights dtype must be bfloat16 or float16 for fused MoE kernel, but got {routing_weights.dtype}"
    )
    assert hidden_states.dtype in [torch.bfloat16, torch.float16], (
        f"hidden_states dtype must be bfloat16 or float16 for fused MoE kernel, but got {hidden_states.dtype}"
    )

    return _fused_lora_moe_forward(
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hidden_states,
        fc1_1_2_weight=fc1_1_2_weight,
        fc2_weight=fc2_weight,
        lora_a_gate=lora_a_gate,
        lora_b_gate=lora_b_gate,
        lora_a_up=lora_a_up,
        lora_b_up=lora_b_up,
        lora_a_down=lora_a_down,
        lora_b_down=lora_b_down,
        lora_scale_gate=lora_scale_gate,
        lora_scale_up=lora_scale_up,
        lora_scale_down=lora_scale_down,
    )


def fused_independent_lora_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    lora_a_gate: torch.Tensor,
    lora_b_gate: torch.Tensor,
    lora_a_up: torch.Tensor,
    lora_b_up: torch.Tensor,
    lora_a_down: torch.Tensor,
    lora_b_down: torch.Tensor,
    lora_scale_gate: float,
    lora_scale_up: float,
    lora_scale_down: float,
):
    """Public dispatcher for the fused MoE forward + independent per-expert seed-style two-LoRA (Mode 1).

    Same shape contract as :func:`fused_lora_moe_forward` except every LoRA
    tensor carries a leading expert dim (``[E, r, ...]`` / ``[E, ..., r]``).
    See
    :func:`veomni.lora.ops.moe_group_gemm.group_gemm_fused_independent_lora_moe_forward`.

    Raises ``NotImplementedError`` if no LoRA-aware fused kernel is bound for
    the active ``moe_implementation`` — callers (see
    :class:`veomni.lora.moe_layers.LoraIndependentExperts`) are expected to
    check this module's ``_fused_independent_lora_moe_forward`` is non-``None``
    before calling.
    """
    if _fused_independent_lora_moe_forward is None:
        raise NotImplementedError(
            "No fused independent MoE-LoRA kernel is bound. Set ops_implementation.moe_implementation "
            "to a backend that ships a LoRA variant ('fused_triton' on GPU, 'fused_npu' on NPU) "
            "or fall back to the eager LoRA forward."
        )

    assert routing_weights.dtype in [torch.bfloat16, torch.float16], (
        f"routing_weights dtype must be bfloat16 or float16 for fused MoE kernel, but got {routing_weights.dtype}"
    )
    assert hidden_states.dtype in [torch.bfloat16, torch.float16], (
        f"hidden_states dtype must be bfloat16 or float16 for fused MoE kernel, but got {hidden_states.dtype}"
    )

    return _fused_independent_lora_moe_forward(
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hidden_states,
        fc1_1_2_weight=fc1_1_2_weight,
        fc2_weight=fc2_weight,
        lora_a_gate=lora_a_gate,
        lora_b_gate=lora_b_gate,
        lora_a_up=lora_a_up,
        lora_b_up=lora_b_up,
        lora_a_down=lora_a_down,
        lora_b_down=lora_b_down,
        lora_scale_gate=lora_scale_gate,
        lora_scale_up=lora_scale_up,
        lora_scale_down=lora_scale_down,
    )
