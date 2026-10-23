"""Verify both 16 GB SM70 devices and execute the candidate's native kernels."""

import json
import time
from pathlib import Path

import torch
import vllm._custom_ops as ops
from torch.nn.attention import SDPBackend, sdpa_kernel


def main():
    assert torch.cuda.device_count() == 2
    result = []
    torch.manual_seed(42)
    for index in range(2):
        with torch.cuda.device(index), torch.inference_mode():
            props = torch.cuda.get_device_properties(index)
            assert (props.major, props.minor) == (7, 0)
            assert 15 * 2**30 <= props.total_memory <= 16 * 2**30
            started = time.monotonic()
            memory = torch.empty(12 * 2**30 // 4, dtype=torch.float32, device="cuda")
            memory.fill_(1.25)
            assert (memory[::1024] == 1.25).all().item()
            del memory
            torch.cuda.empty_cache()
            a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
            b = torch.randn_like(a)
            expected = a.float() @ b.float()
            for _ in range(16):
                actual = a @ b
                torch.testing.assert_close(
                    actual.float(), expected, atol=0.125, rtol=0.002
                )
            x = torch.randn(16, 256, device="cuda", dtype=torch.float16)
            w = torch.ones(256, device="cuda", dtype=torch.float16)
            output = torch.empty_like(x)
            ops.rms_norm(output, x, w, 1e-6)
            expected = x.float() * torch.rsqrt(
                x.float().square().mean(-1, keepdim=True) + 1e-6
            )
            torch.testing.assert_close(output.float(), expected, atol=0.005, rtol=0.005)
            q = torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.float16)
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                output = torch.nn.functional.scaled_dot_product_attention(
                    q, q, q, is_causal=True
                )
            assert torch.isfinite(output).all().item()
            torch.cuda.synchronize()
            result.append(
                dict(
                    index=index,
                    name=props.name,
                    memory_bytes=props.total_memory,
                    capability=[props.major, props.minor],
                    checked_allocation_bytes=12 * 2**30,
                    elapsed_s=time.monotonic() - started,
                    passed=True,
                )
            )
    Path("/workspace/results/hardware-check.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
