# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-one FP8 activation quantization for Qwen MTP draft heads."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


_FP8_MAX = 448.0
_FLOAT32_TINY = torch.finfo(torch.float32).tiny


@triton.jit
def _quantize_mtp_head_input_fp8_kernel(
    input_ptr,
    quant_out_ptr,
    scale_out_ptr,
    num_elements: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.max(tl.abs(values), axis=0) * (1.0 / FP8_MAX)
    scale = tl.maximum(scale, 1.1754943508222875e-38)

    # PyTorch treats the 0D FP32 scale as a wrapped scalar for BF16 division.
    scale_bf16 = scale.to(tl.bfloat16).to(tl.float32)
    scaled = tl.div_rn(values, scale_bf16).to(tl.bfloat16)
    quantized = tl.clamp(scaled, -FP8_MAX, FP8_MAX).to(
        quant_out_ptr.dtype.element_ty
    )
    tl.store(quant_out_ptr + offsets, quantized, mask=mask)
    tl.store(scale_out_ptr, scale)


def quantize_mtp_head_input_fp8(
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the private MTP head's tensor-wise dynamic E4M3 quantization."""
    if (
        current_platform.is_cuda()
        and hidden_states.is_contiguous()
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.numel() == 4096
    ):
        hidden_fp8 = torch.empty_like(hidden_states, dtype=torch.float8_e4m3fn)
        hidden_scale = torch.empty((), device=hidden_states.device, dtype=torch.float32)
        _quantize_mtp_head_input_fp8_kernel[(1,)](
            hidden_states,
            hidden_fp8,
            hidden_scale,
            num_elements=hidden_states.numel(),
            FP8_MAX=_FP8_MAX,
            BLOCK_SIZE=4096,
            num_warps=8,
            num_stages=1,
        )
        return hidden_fp8, hidden_scale

    hidden_scale = (
        hidden_states.abs().max().float() / _FP8_MAX
    ).clamp_min(_FLOAT32_TINY)
    hidden_fp8 = (hidden_states / hidden_scale).clamp(-_FP8_MAX, _FP8_MAX).to(
        torch.float8_e4m3fn
    )
    return hidden_fp8, hidden_scale