# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""TorchAO MXFP8 grouped GEMM backend for MoE experiments.

This module keeps the scale layouts and SM100 kernels delegated to
``torchao.prototype.moe_training`` instead of duplicating those conventions in
VeOmni.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor


MXFP8_BLOCK_SIZE = 32


def is_torch_scaled_grouped_gemm_available() -> bool:
    """Return whether TorchAO's SM100 MXFP8 grouped GEMM path is importable."""
    try:
        from torchao.prototype.moe_training import mxfp8_grouped_mm

        return bool(getattr(mxfp8_grouped_mm, "_SM100_KERNELS_AVAILABLE", False))
    except Exception:
        return False


def _torchao_imports():
    try:
        from torchao.prototype.moe_training.mxfp8_grouped_mm import (
            _compute_wgrad,
            _to_mxfp8_then_scaled_grouped_mm,
        )
        from torchao.prototype.moe_training.utils import pad_token_groups
        from torchao.prototype.mx_formats.config import ScaleCalculationMode
        from torchao.quantization.quantize_.common import KernelPreference
    except Exception as exc:
        raise RuntimeError(
            "TorchAO MXFP8 grouped GEMM requires torchao with "
            "torchao.prototype.moe_training support. "
            "On Blackwell, install Open-VeOmni with the gpu extra, which pins "
            "torch>=2.11.0+cu130 and torchao==0.17.0+cu130 from the PyTorch index."
        ) from exc
    return _compute_wgrad, _to_mxfp8_then_scaled_grouped_mm, pad_token_groups, ScaleCalculationMode, KernelPreference


def _to_group_end_offsets(cu_seqlens: Tensor) -> Tensor:
    return cu_seqlens[1:].contiguous().to(torch.int32)


def _use_hp_wgrad() -> bool:
    return os.getenv("VEOMNI_TORCHAO_MXFP8_WGRAD_WITH_HP", "1").lower() in {"1", "true", "yes", "on"}


def _to_torchao_weight_t_layout(weight_t: Tensor) -> Tensor:
    if weight_t.transpose(-2, -1).is_contiguous():
        return weight_t
    return weight_t.transpose(-2, -1).contiguous().transpose(-2, -1)


def torch_scaled_grouped_varlen_m_gemm(
    a: Tensor,
    b_lkn: Tensor,
    cu_seqlens_m: Tensor,
    *,
    a_idx: Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> Tensor:
    """Compute per-expert ``A @ B`` using TorchAO SM100 MXFP8 grouped GEMM."""
    if a_idx is not None:
        a = a[a_idx.long()]
    a = a.contiguous()
    b_lkn = _to_torchao_weight_t_layout(b_lkn)

    out_dtype = a.dtype if out_dtype is None else out_dtype
    _, to_mxfp8_grouped_mm, _, ScaleCalculationMode, KernelPreference = _torchao_imports()
    return to_mxfp8_grouped_mm(
        a,
        b_lkn,
        offs=_to_group_end_offsets(cu_seqlens_m),
        block_size=MXFP8_BLOCK_SIZE,
        out_dtype=out_dtype,
        kernel_preference=KernelPreference.AUTO,
        wgrad_with_hp=_use_hp_wgrad(),
        scale_calculation_mode=ScaleCalculationMode.RCEIL,
        pad_token_groups_for_grouped_mm=True,
    )


def torch_scaled_grouped_varlen_k_gemm(
    a: Tensor,
    b: Tensor,
    cu_seqlens_k: Tensor,
    *,
    out_dtype: torch.dtype | None = None,
) -> Tensor:
    """Compute per-expert wgrad-like ``A_i @ B_i`` with TorchAO MXFP8.

    Args:
        a: ``(m, total_k)`` tensor split by ``cu_seqlens_k`` along K.
        b: ``(total_k, n)`` tensor split by ``cu_seqlens_k`` along rows.
    Returns:
        ``(num_experts, m, n)``, matching ``quack.gemm(..., cu_seqlens_k=...)``.
    """
    out_dtype = a.dtype if out_dtype is None else out_dtype
    compute_wgrad, _, pad_token_groups, ScaleCalculationMode, KernelPreference = _torchao_imports()
    group_end_offsets = _to_group_end_offsets(cu_seqlens_k)

    grad_output = a.transpose(-2, -1).contiguous()
    input_act = b.contiguous()
    padded_grad_output, _, padded_group_end_offsets = pad_token_groups(
        grad_output,
        group_end_offsets,
        alignment_size=MXFP8_BLOCK_SIZE,
        kernel_preference=KernelPreference.AUTO,
    )
    padded_input_act, _, _ = pad_token_groups(
        input_act,
        group_end_offsets,
        alignment_size=MXFP8_BLOCK_SIZE,
        kernel_preference=KernelPreference.AUTO,
    )

    grad_weight_t = compute_wgrad(
        padded_grad_output,
        padded_input_act,
        padded_group_end_offsets,
        MXFP8_BLOCK_SIZE,
        out_dtype,
        ScaleCalculationMode.RCEIL,
        _use_hp_wgrad(),
        KernelPreference.AUTO,
    )
    return grad_weight_t.transpose(-2, -1).contiguous()


def torch_scaled_grouped_gemm(
    a: Tensor,
    b: Tensor,
    *,
    cu_seqlens_m: Tensor | None = None,
    cu_seqlens_k: Tensor | None = None,
    A_idx: Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> Tensor:
    """Dispatch grouped GEMM by whether the ragged dimension is M or K."""
    if (cu_seqlens_m is None) == (cu_seqlens_k is None):
        raise ValueError("TorchAO MXFP8 MoE GEMM requires exactly one of cu_seqlens_m or cu_seqlens_k")
    if cu_seqlens_m is not None:
        return torch_scaled_grouped_varlen_m_gemm(a, b, cu_seqlens_m, a_idx=A_idx, out_dtype=out_dtype)
    if A_idx is not None:
        raise NotImplementedError("TorchAO MXFP8 varlen-K GEMM does not support A_idx")
    return torch_scaled_grouped_varlen_k_gemm(a, b, cu_seqlens_k, out_dtype=out_dtype)
