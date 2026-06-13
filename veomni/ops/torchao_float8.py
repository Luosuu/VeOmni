# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import torch
import torch.nn as nn

from ..arguments import VeOmniArguments
from ..utils import logging


logger = logging.get_logger(__name__)
MXFP8_RECIPE_NAMES = {"mxfp8", "mxfp8_with_gw_hp"}
MXFP8_BLOCK_SIZE = 32


class TorchAOMXFP8Linear(nn.Linear):
    """Dense Linear backed by TorchAO prototype MXFP8 matmul kernels."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        wgrad_with_hp: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        _, ScaleCalculationMode, KernelPreference = _torchao_mxfp8_imports()
        self.kernel_preference = KernelPreference.AUTO
        self.scale_calculation_mode = ScaleCalculationMode.RCEIL
        self.wgrad_with_hp = wgrad_with_hp

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, wgrad_with_hp: bool) -> TorchAOMXFP8Linear:
        new_linear = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            wgrad_with_hp=wgrad_with_hp,
        )
        new_linear.weight = linear.weight
        new_linear.bias = linear.bias
        return new_linear

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        to_mxfp8_scaled_mm, _, _ = _torchao_mxfp8_imports()
        output = to_mxfp8_scaled_mm(
            input,
            self.weight,
            self.kernel_preference,
            self.scale_calculation_mode,
            self.wgrad_with_hp,
        )
        if self.bias is not None:
            output = output + self.bias
        return output


def _torchao_mxfp8_imports():
    try:
        from torchao.prototype.mx_formats.config import ScaleCalculationMode
        from torchao.prototype.mx_formats.mx_linear import _to_mxfp8_then_scaled_mm
        from torchao.quantization.quantize_.common import KernelPreference
    except Exception as exc:
        raise ImportError(
            "TorchAO dense MXFP8 Linear requires torchao with torchao.prototype.mx_formats support. "
            "On Blackwell, install Open-VeOmni with the gpu extra, which pins "
            "torch>=2.11.0+cu130 and torchao==0.17.0+cu130 from the PyTorch index."
        ) from exc
    return _to_mxfp8_then_scaled_mm, ScaleCalculationMode, KernelPreference


def _get_float8_filter(args: VeOmniArguments):
    filter_fqns = tuple(args.train.torchao_float8_filter_fqns)
    recipe_name = args.train.torchao_float8_recipe_name

    auto_filter = None
    if args.train.torchao_float8_auto_filter_small_kn and recipe_name not in MXFP8_RECIPE_NAMES:
        try:
            from torchao.float8 import _auto_filter_for_recipe

            auto_filter = _auto_filter_for_recipe(recipe_name, filter_fqns=list(filter_fqns))
        except ImportError:
            logger.warning_rank0(
                "torchao _auto_filter_for_recipe is unavailable; falling back to shape/FQN filtering."
            )

    def module_filter_fn(module: nn.Module, fqn: str) -> bool:
        if any(pattern in fqn for pattern in filter_fqns):
            return False
        if auto_filter is not None:
            return auto_filter(module, fqn)
        if isinstance(module, nn.Linear):
            return module.in_features % 16 == 0 and module.out_features % 16 == 0
        return True

    return module_filter_fn


def _is_mxfp8_recipe(recipe_name: str) -> bool:
    return recipe_name in MXFP8_RECIPE_NAMES


def _is_mxfp8_linear_candidate(module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    return module.in_features % MXFP8_BLOCK_SIZE == 0 and module.out_features % MXFP8_BLOCK_SIZE == 0


def _convert_to_mxfp8_training_with_fqns(model: nn.Module, args: VeOmniArguments) -> int:
    filter_fn = _get_float8_filter(args)
    wgrad_with_hp = args.train.torchao_float8_recipe_name == "mxfp8_with_gw_hp"
    converted = 0

    def convert_children(module: nn.Module, prefix: str = "") -> None:
        nonlocal converted
        for child_name, child in list(module.named_children()):
            fqn = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and not isinstance(child, TorchAOMXFP8Linear):
                if filter_fn(child, fqn) and _is_mxfp8_linear_candidate(child):
                    setattr(module, child_name, TorchAOMXFP8Linear.from_linear(child, wgrad_with_hp=wgrad_with_hp))
                    converted += 1
                continue
            convert_children(child, fqn)

    convert_children(model)
    return converted


def apply_torchao_float8_training(model: nn.Module, args: VeOmniArguments) -> nn.Module:
    """Convert eligible ``nn.Linear`` modules to TorchAO low-precision Linear before FSDP wrapping."""
    if not getattr(args.train, "enable_torchao_float8", False):
        return model

    recipe_name = args.train.torchao_float8_recipe_name
    linear_count = sum(isinstance(module, nn.Linear) for module in model.modules())
    if _is_mxfp8_recipe(recipe_name):
        _torchao_mxfp8_imports()
        converted_count = _convert_to_mxfp8_training_with_fqns(model, args)
        logger.info_rank0(
            f"Enabled torchao dense MXFP8 training with recipe={recipe_name!r}: "
            f"converted {converted_count}/{linear_count} nn.Linear modules."
        )
        return model

    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        from torchao.float8.float8_linear import Float8Linear
    except ImportError as exc:
        raise ImportError(
            "train.enable_torchao_float8=True requires torchao. Install Open-VeOmni with the gpu extra."
        ) from exc

    if hasattr(Float8LinearConfig, "from_recipe_name"):
        config = Float8LinearConfig.from_recipe_name(recipe_name)
    else:
        if recipe_name != "tensorwise":
            logger.warning_rank0(
                "Installed torchao does not support Float8LinearConfig.from_recipe_name; "
                "falling back to torchao's default float8 config."
            )
        config = Float8LinearConfig()

    model = convert_to_float8_training(model, config=config, module_filter_fn=_get_float8_filter(args))
    float8_count = sum(isinstance(module, Float8Linear) for module in model.modules())
    logger.info_rank0(
        f"Enabled torchao float8 training with recipe={recipe_name!r}: "
        f"converted {float8_count}/{linear_count} nn.Linear modules."
    )
    return model
