import importlib.util
from pathlib import Path

import torch


def _load_torch_scaled_grouped_gemm():
    module_path = Path(__file__).parents[2] / "veomni/ops/kernels/moe/torch_scaled_grouped_gemm.py"
    spec = importlib.util.spec_from_file_location("torch_scaled_grouped_gemm", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


torch_scaled_grouped_gemm = _load_torch_scaled_grouped_gemm()


def test_torch_scaled_grouped_gemm_dispatches_varlen_m(monkeypatch):
    calls = []

    def fake_varlen_m(a, b, cu_seqlens_m, *, a_idx=None, out_dtype=None):
        calls.append(("m", a, b, cu_seqlens_m, a_idx, out_dtype))
        return torch.empty(1)

    monkeypatch.setattr(torch_scaled_grouped_gemm, "torch_scaled_grouped_varlen_m_gemm", fake_varlen_m)

    a = torch.empty(4, 32)
    b = torch.empty(2, 32, 16)
    cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    a_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    torch_scaled_grouped_gemm.torch_scaled_grouped_gemm(a, b, cu_seqlens_m=cu, A_idx=a_idx, out_dtype=torch.bfloat16)

    assert calls == [("m", a, b, cu, a_idx, torch.bfloat16)]


def test_torch_scaled_grouped_gemm_dispatches_varlen_k(monkeypatch):
    calls = []

    def fake_varlen_k(a, b, cu_seqlens_k, *, out_dtype=None):
        calls.append(("k", a, b, cu_seqlens_k, out_dtype))
        return torch.empty(1)

    monkeypatch.setattr(torch_scaled_grouped_gemm, "torch_scaled_grouped_varlen_k_gemm", fake_varlen_k)

    a = torch.empty(16, 64)
    b = torch.empty(64, 32)
    cu = torch.tensor([0, 32, 64], dtype=torch.int32)
    torch_scaled_grouped_gemm.torch_scaled_grouped_gemm(a, b, cu_seqlens_k=cu, out_dtype=torch.float16)

    assert calls == [("k", a, b, cu, torch.float16)]


def test_varlen_k_pads_ragged_token_groups(monkeypatch):
    seen_offsets = []

    class FakeKernelPreference:
        AUTO = object()

    class FakeScaleCalculationMode:
        RCEIL = object()

    def fake_compute_wgrad(
        grad_output,
        input_act,
        group_end_offsets,
        block_size,
        out_dtype,
        scale_calculation_mode,
        wgrad_with_hp,
        kernel_preference,
    ):
        seen_offsets.append(group_end_offsets.clone())
        assert grad_output.shape == (64, 8)
        assert input_act.shape == (64, 4)
        assert block_size == 32
        assert wgrad_with_hp
        return torch.zeros(2, 4, 8, dtype=out_dtype)

    def fake_pad_token_groups(x, offs, *, alignment_size, kernel_preference):
        assert alignment_size == 32
        counts = torch.diff(torch.cat([torch.zeros(1, dtype=offs.dtype), offs.cpu()]))
        padded_counts = ((counts + 31) // 32) * 32
        padded = torch.zeros(int(padded_counts.sum()), x.shape[1], dtype=x.dtype)
        start = 0
        padded_start = 0
        for count, padded_count in zip(counts.tolist(), padded_counts.tolist()):
            padded[padded_start : padded_start + count] = x[start : start + count]
            start += count
            padded_start += padded_count
        return padded, None, torch.cumsum(padded_counts.to(offs.dtype), dim=0)

    monkeypatch.setattr(
        torch_scaled_grouped_gemm,
        "_torchao_imports",
        lambda: (
            fake_compute_wgrad,
            None,
            fake_pad_token_groups,
            FakeScaleCalculationMode,
            FakeKernelPreference,
        ),
    )

    a = torch.empty(8, 33)
    b = torch.empty(33, 4)
    cu = torch.tensor([0, 1, 33], dtype=torch.int32)

    out = torch_scaled_grouped_gemm.torch_scaled_grouped_varlen_k_gemm(a, b, cu, out_dtype=torch.bfloat16)

    assert out.shape == (2, 8, 4)
    assert seen_offsets[0].tolist() == [32, 64]


def test_varlen_m_passes_more_than_32_groups_to_torchao(monkeypatch):
    calls = []

    class FakeKernelPreference:
        AUTO = object()

    class FakeScaleCalculationMode:
        RCEIL = object()

    def fake_to_mxfp8_grouped_mm(a, b, *, offs, block_size, out_dtype, **kwargs):
        calls.append((a.shape, b.shape, offs.tolist(), block_size, out_dtype))
        return torch.full((a.shape[0], b.shape[-1]), 1, dtype=out_dtype)

    monkeypatch.setattr(
        torch_scaled_grouped_gemm,
        "_torchao_imports",
        lambda: (None, fake_to_mxfp8_grouped_mm, None, FakeScaleCalculationMode, FakeKernelPreference),
    )

    num_groups = 34
    counts = torch.ones(num_groups, dtype=torch.int32)
    cu = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(counts, dim=0)])
    a = torch.empty(int(cu[-1].item()), 8)
    b = torch.empty(num_groups, 8, 4)

    out = torch_scaled_grouped_gemm.torch_scaled_grouped_varlen_m_gemm(a, b, cu, out_dtype=torch.bfloat16)

    assert out.shape == (num_groups, 4)
    assert calls == [(torch.Size([34, 8]), torch.Size([34, 8, 4]), list(range(1, 35)), 32, torch.bfloat16)]


def test_varlen_k_passes_more_than_32_groups_to_torchao(monkeypatch):
    calls = []

    class FakeKernelPreference:
        AUTO = object()

    class FakeScaleCalculationMode:
        RCEIL = object()

    def fake_compute_wgrad(grad_output, input_act, group_end_offsets, block_size, out_dtype, *args):
        calls.append((grad_output.shape, input_act.shape, group_end_offsets.tolist(), block_size, out_dtype))
        return torch.full((group_end_offsets.numel(), input_act.shape[1], grad_output.shape[1]), 1, dtype=out_dtype)

    def fake_pad_token_groups(x, offs, *, alignment_size, kernel_preference):
        return x, None, offs

    monkeypatch.setattr(
        torch_scaled_grouped_gemm,
        "_torchao_imports",
        lambda: (
            fake_compute_wgrad,
            None,
            fake_pad_token_groups,
            FakeScaleCalculationMode,
            FakeKernelPreference,
        ),
    )

    num_groups = 34
    counts = torch.ones(num_groups, dtype=torch.int32)
    cu = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(counts, dim=0)])
    a = torch.empty(8, int(cu[-1].item()))
    b = torch.empty(int(cu[-1].item()), 4)

    out = torch_scaled_grouped_gemm.torch_scaled_grouped_varlen_k_gemm(a, b, cu, out_dtype=torch.bfloat16)

    assert out.shape == (num_groups, 8, 4)
    assert calls == [(torch.Size([34, 8]), torch.Size([34, 4]), list(range(1, 35)), 32, torch.bfloat16)]


def test_torchao_import_error_is_actionable(monkeypatch):
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("torchao"):
            raise ImportError("missing torchao")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    try:
        torch_scaled_grouped_gemm._torchao_imports()
    except RuntimeError as exc:
        assert "requires torchao" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
