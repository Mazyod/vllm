"""Execute the legacy image's required kernels on each 16 GB V100.

Run inside the candidate image before downloading checkpoints. A package import
or architecture listing alone cannot establish executable SM70 coverage.
"""

import importlib.metadata as metadata
import json
import sys

import torch
import torch.nn.functional as F
from vllm import _custom_ops as ops
from xformers.ops import memory_efficient_attention


def main() -> int:
    torch.manual_seed(7)
    report = {
        "packages": {
            p: metadata.version(p)
            for p in ("torch", "vllm", "xformers", "transformers")
        },
        "cuda_runtime": torch.version.cuda,
        "torch_architectures": torch.cuda.get_arch_list(),
        "devices": [],
    }
    passed = torch.cuda.device_count() == 2
    for index in range(torch.cuda.device_count()):
        torch.cuda.set_device(index)
        prop = torch.cuda.get_device_properties(index)
        device = {
            "index": index,
            "name": prop.name,
            "memory_bytes": prop.total_memory,
            "capability": list(torch.cuda.get_device_capability(index)),
            "checks": {},
        }
        passed &= device["capability"] == [7, 0] and 15e9 < prop.total_memory < 17.2e9
        for dtype in (torch.float16, torch.float32):
            try:
                x = torch.ones((32, 32), device="cuda", dtype=dtype)
                torch.testing.assert_close(x @ x, torch.full_like(x, 32))
                x = torch.randn((8, 128), device="cuda", dtype=dtype)
                weight = torch.ones(128, device="cuda", dtype=dtype)
                output = torch.empty_like(x)
                ops.rms_norm(output, x, weight, 1e-5)
                reference = x.float() * torch.rsqrt(
                    x.float().square().mean(-1, keepdim=True) + 1e-5
                )
                torch.testing.assert_close(
                    output.float(), reference, rtol=0.002, atol=0.002
                )
                q, k, v = [
                    torch.randn((1, 32, 4, 64), device="cuda", dtype=dtype)
                    for _ in range(3)
                ]
                actual = memory_efficient_attention(q, k, v)
                reference = F.scaled_dot_product_attention(
                    q.float().transpose(1, 2),
                    k.float().transpose(1, 2),
                    v.float().transpose(1, 2),
                ).transpose(1, 2)
                torch.testing.assert_close(
                    actual.float(), reference, rtol=0.005, atol=0.005
                )
                torch.cuda.synchronize()
                device["checks"][str(dtype)] = {"passed": True}
            except Exception as exc:
                device["checks"][str(dtype)] = {"passed": False, "error": str(exc)}
                passed = False
        report["devices"].append(device)
    report["passed"] = bool(passed)
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
