import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn


def _load_torchao_float8(monkeypatch):
    veomni = types.ModuleType("veomni")
    veomni.__path__ = []
    ops = types.ModuleType("veomni.ops")
    ops.__path__ = []
    arguments = types.ModuleType("veomni.arguments")
    arguments.VeOmniArguments = object

    class FakeLogger:
        def warning_rank0(self, *args, **kwargs):
            pass

        def info_rank0(self, *args, **kwargs):
            pass

    logging = types.ModuleType("veomni.utils.logging")
    logging.get_logger = lambda name: FakeLogger()
    utils = types.ModuleType("veomni.utils")
    utils.logging = logging

    monkeypatch.setitem(sys.modules, "veomni", veomni)
    monkeypatch.setitem(sys.modules, "veomni.ops", ops)
    monkeypatch.setitem(sys.modules, "veomni.arguments", arguments)
    monkeypatch.setitem(sys.modules, "veomni.utils", utils)
    monkeypatch.setitem(sys.modules, "veomni.utils.logging", logging)

    module_path = Path(__file__).parents[2] / "veomni/ops/torchao_float8.py"
    spec = importlib.util.spec_from_file_location("veomni.ops.torchao_float8", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@dataclass
class _TrainArgs:
    enable_torchao_float8: bool = False
    torchao_float8_recipe_name: str = "rowwise"
    torchao_float8_filter_fqns: list[str] = field(
        default_factory=lambda: ["lm_head", "lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"]
    )
    torchao_float8_auto_filter_small_kn: bool = True


@dataclass
class _Args:
    train: _TrainArgs = field(default_factory=_TrainArgs)


def test_torchao_float8_disabled_returns_same_model(monkeypatch):
    torchao_float8 = _load_torchao_float8(monkeypatch)
    model = nn.Sequential(nn.Linear(16, 16))

    assert torchao_float8.apply_torchao_float8_training(model, _Args()) is model


def test_torchao_float8_conversion_uses_filter(monkeypatch):
    torchao_float8 = _load_torchao_float8(monkeypatch)
    calls = []

    class FakeFloat8Linear(nn.Module):
        pass

    class FakeFloat8LinearConfig:
        @staticmethod
        def from_recipe_name(recipe_name):
            return {"recipe": recipe_name}

    def fake_convert_to_float8_training(model, *, config, module_filter_fn):
        calls.append(("config", config))
        for fqn, module in model.named_modules():
            if fqn and isinstance(module, nn.Linear):
                calls.append((fqn, module_filter_fn(module, fqn)))
        return model

    torchao = types.ModuleType("torchao")
    float8 = types.ModuleType("torchao.float8")
    float8.Float8LinearConfig = FakeFloat8LinearConfig
    float8.convert_to_float8_training = fake_convert_to_float8_training
    float8._auto_filter_for_recipe = lambda recipe_name, filter_fqns: (
        lambda module, fqn: not any(pattern in fqn for pattern in filter_fqns)
    )
    float8_linear = types.ModuleType("torchao.float8.float8_linear")
    float8_linear.Float8Linear = FakeFloat8Linear

    monkeypatch.setitem(sys.modules, "torchao", torchao)
    monkeypatch.setitem(sys.modules, "torchao.float8", float8)
    monkeypatch.setitem(sys.modules, "torchao.float8.float8_linear", float8_linear)

    model = nn.Module()
    model.good = nn.Linear(16, 16)
    model.lm_head = nn.Linear(16, 16)
    args = _Args(_TrainArgs(enable_torchao_float8=True, torchao_float8_filter_fqns=["lm_head"]))

    assert torchao_float8.apply_torchao_float8_training(model, args) is model
    assert ("config", {"recipe": "rowwise"}) in calls
    assert ("good", True) in calls
    assert ("lm_head", False) in calls


def test_torchao_float8_default_filter_skips_lora_adapters(monkeypatch):
    torchao_float8 = _load_torchao_float8(monkeypatch)
    filter_fn = torchao_float8._get_float8_filter(_Args())

    assert filter_fn(nn.Linear(16, 16), "model.layers.0.self_attn.q_proj.base_layer")
    assert not filter_fn(nn.Linear(16, 16), "model.layers.0.self_attn.q_proj.lora_A.default")
    assert not filter_fn(nn.Linear(16, 16), "model.layers.0.self_attn.q_proj.lora_B.default")


def test_torchao_mxfp8_conversion_uses_block32_and_filter(monkeypatch):
    torchao_float8 = _load_torchao_float8(monkeypatch)
    calls = []

    def fake_to_mxfp8_scaled_mm(input, weight, kernel_preference, scale_calculation_mode, wgrad_with_hp):
        calls.append((kernel_preference, scale_calculation_mode, wgrad_with_hp))
        return input @ weight.t()

    class FakeKernelPreference:
        AUTO = "auto"

    class FakeScaleCalculationMode:
        RCEIL = "rceil"

    torchao = types.ModuleType("torchao")
    prototype = types.ModuleType("torchao.prototype")
    mx_formats = types.ModuleType("torchao.prototype.mx_formats")
    mx_config = types.ModuleType("torchao.prototype.mx_formats.config")
    mx_config.ScaleCalculationMode = FakeScaleCalculationMode
    mx_linear = types.ModuleType("torchao.prototype.mx_formats.mx_linear")
    mx_linear._to_mxfp8_then_scaled_mm = fake_to_mxfp8_scaled_mm
    quantization = types.ModuleType("torchao.quantization")
    quantize_ = types.ModuleType("torchao.quantization.quantize_")
    common = types.ModuleType("torchao.quantization.quantize_.common")
    common.KernelPreference = FakeKernelPreference

    monkeypatch.setitem(sys.modules, "torchao", torchao)
    monkeypatch.setitem(sys.modules, "torchao.prototype", prototype)
    monkeypatch.setitem(sys.modules, "torchao.prototype.mx_formats", mx_formats)
    monkeypatch.setitem(sys.modules, "torchao.prototype.mx_formats.config", mx_config)
    monkeypatch.setitem(sys.modules, "torchao.prototype.mx_formats.mx_linear", mx_linear)
    monkeypatch.setitem(sys.modules, "torchao.quantization", quantization)
    monkeypatch.setitem(sys.modules, "torchao.quantization.quantize_", quantize_)
    monkeypatch.setitem(sys.modules, "torchao.quantization.quantize_.common", common)

    model = nn.Module()
    model.good = nn.Linear(32, 64, bias=False)
    model.small = nn.Linear(16, 16, bias=False)
    model.lm_head = nn.Linear(32, 32, bias=False)
    args = _Args(
        _TrainArgs(
            enable_torchao_float8=True,
            torchao_float8_recipe_name="mxfp8_with_gw_hp",
            torchao_float8_filter_fqns=["lm_head"],
        )
    )

    assert torchao_float8.apply_torchao_float8_training(model, args) is model
    assert isinstance(model.good, torchao_float8.TorchAOMXFP8Linear)
    assert isinstance(model.small, nn.Linear)
    assert not isinstance(model.small, torchao_float8.TorchAOMXFP8Linear)
    assert isinstance(model.lm_head, nn.Linear)
    assert not isinstance(model.lm_head, torchao_float8.TorchAOMXFP8Linear)

    output = model.good(torch.ones(2, 32))
    assert output.shape == (2, 64)
    assert calls == [("auto", "rceil", True)]
