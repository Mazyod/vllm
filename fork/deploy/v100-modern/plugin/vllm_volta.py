"""v0.32-only pooling attention; retain model math and full input contexts.

Retire when upstream supplies a comparable SM70 pooling backend.
"""

# ruff: noqa: INP001
# Installed as a setuptools py-module, not an implicit namespace package.

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from vllm.config import get_current_vllm_config
from vllm.v1.attention.backends.flex_attention import (
    FlexAttentionBackend,
    FlexAttentionImpl,
)


class SDPAPoolingBackend(FlexAttentionBackend):
    """Use native vLLM metadata/cache management with packaged Torch SDPA."""

    supported_dtypes = [torch.float16]

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return SDPAPoolingImpl

    @classmethod
    def supports_sliding_window(cls):
        return False

    @classmethod
    def supports_rswa(cls):
        return False

    @classmethod
    def supports_mm_prefix(cls):
        return False


class SDPAPoolingImpl(FlexAttentionImpl):
    """Full-prefill pooling only, with original causal/bidirectional semantics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = get_current_vllm_config()
        if config.model_config.runner_type != "pooling":
            raise ValueError("VOLTA_SDPA is restricted to full-prefill pooling")
        if self.sliding_window is not None or self.out_transform is not None:
            raise ValueError("VOLTA_SDPA does not implement window/epilogue transforms")

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if output_scale is not None or output_block_scale is not None:
            raise ValueError("VOLTA_SDPA does not support quantized output")
        if attn_metadata is None:
            return output.zero_()
        if query.dtype != torch.float16:
            raise ValueError("VOLTA_SDPA requires FP16 inputs")
        if (
            getattr(layer, "logical_mask_mod", None) is not None
            or getattr(layer, "block_sparsity_hint", None) is not None
            or getattr(attn_metadata, "mm_prefix_range", None) is not None
        ):
            raise ValueError("VOLTA_SDPA refuses custom attention masks")
        count = attn_metadata.num_reqs
        boundaries = attn_metadata.query_start_loc_cpu[: count + 1].tolist()
        lengths = attn_metadata.seq_lens[:count].tolist()
        if any(
            end - start != length
            for start, end, length in zip(boundaries[:-1], boundaries[1:], lengths)
        ):
            raise ValueError(
                "VOLTA_SDPA requires complete inputs without cached prefixes"
            )
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                q = query[start:end].transpose(0, 1).unsqueeze(0)
                k = key[start:end].repeat_interleave(self.num_queries_per_kv, dim=1)
                v = value[start:end].repeat_interleave(self.num_queries_per_kv, dim=1)
                result = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k.transpose(0, 1).unsqueeze(0),
                    v.transpose(0, 1).unsqueeze(0),
                    is_causal=attn_metadata.causal,
                    scale=self.scale,
                    dropout_p=0.0,
                )
                output[start:end].copy_(result.squeeze(0).transpose(0, 1))
        return output


def register():
    """Register through vLLM's supported plugin APIs; no runtime monkeypatches."""
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    # The existing enum survives spawned-worker deserialization, which happens
    # before plugins can register a dynamically named enum in that worker.
    register_backend(AttentionBackendEnum.CUSTOM, "vllm_volta.SDPAPoolingBackend")
