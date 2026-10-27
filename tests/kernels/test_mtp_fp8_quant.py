# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.mtp_fp8_quant import quantize_mtp_head_input_fp8
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="MTP FP8 quantization Triton kernel requires CUDA",
)
@pytest.mark.parametrize("seed", [0, 13, 99])
@torch.inference_mode()
def test_mtp_fp8_quant_matches_pytorch(seed: int):
    device = torch.device("cuda", torch.accelerator.current_device_index())
    set_random_seed(seed)
    hidden_states = torch.randn(
        (1, 4096), dtype=torch.bfloat16, device=device
    )

    fp8_max = 448.0
    expected_scale = (
        hidden_states.abs().max().float() / fp8_max
    ).clamp_min(torch.finfo(torch.float32).tiny)
    expected_fp8 = (hidden_states / expected_scale).clamp(
        -fp8_max, fp8_max
    ).to(torch.float8_e4m3fn)

    actual_fp8, actual_scale = quantize_mtp_head_input_fp8(hidden_states)

    torch.testing.assert_close(actual_scale, expected_scale, atol=0, rtol=0)
    torch.testing.assert_close(actual_fp8, expected_fp8, atol=0, rtol=0)