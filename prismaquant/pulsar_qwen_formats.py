"""What the pulsar Qwen3.8-Flash-Next lane can emit -- the `pulsar_qwen` serving profile's exporter declaration.

The exporter is pulsar's direct builder (`tools/container`, out of tree), which writes exactly the formats the engine has
(or is building) kernels for:

  * EXL3_K2 .. EXL3_K5 -- exllamav3's trellis (EXL3) format, bit-compatible with exllamav3 (pulsar L245): K bits per
    weight in 16x16 tiles, a 128-blockwise Hadamard on BOTH dims plus fp16 input/output sign-and-scale vectors
    (suh [in], svh [out]).  So the size is K * in * out + 16 * (in + out) bits per matrix, and the format exists only
    for in % 128 == 0 and out % 128 == 0 (exllamav3 asserts both; `exl3_bits_for_shape` refuses the rest, which is
    the registry's shape gate).
  * MXFP8_E4M3 -- the 8-bit tier (pulsar mxfp8_lt: E4M3 elements, E8M0 scale per 32).
  * BF16 -- passthrough (embedding, output head, norms, small tensors).

EXL3 has NO in-tree codec: prismaquant cannot render these rungs, so their costs come from a measured cost model
(pulsar-notes research/l251/alloc: sampled `quantize_exl3` runs with real Hessians, scaled per tensor class and K),
not from `quantize_dequantize` -- which refuses rather than pretending.
"""
from __future__ import annotations

from fractions import Fraction

from . import format_registry as fr

EXL3_KS = (2, 3, 4, 5)


def exl3_bits_for_shape(K: int):
    def bits(shape: tuple[int, ...]) -> Fraction:
        if len(shape) not in (2, 3):
            raise ValueError(f"EXL3 needs a 2-D matrix or a 3-D expert stack, got {shape}")
        stack = shape[0] if len(shape) == 3 else 1
        out_f, in_f = int(shape[-2]), int(shape[-1])
        if in_f % 128 or out_f % 128:
            raise ValueError(f"EXL3 needs in and out multiples of 128 (128-blockwise Hadamard), got {shape}")
        return Fraction(stack * (K * in_f * out_f + 16 * (in_f + out_f)))
    return bits


def _no_codec(name: str):
    def refuse(_x):
        raise NotImplementedError(
            f"{name}: no in-tree EXL3 codec; price it from the sampled cost model (research/l251/alloc)")
    return refuse


def _register() -> None:
    for K in EXL3_KS:
        name = f"EXL3_K{K}"
        if name in fr.REGISTRY:
            continue
        fr.register_format(fr.FormatSpec(
            name=name, weight_bits=K, group_size=0, scale_bits=0,
            scale_dtype_name="fp16", weight_element_dtype=f"exl3_trellis_k{K}",
            act_bits=None, family="exl3", min_capability_sm=80,
            quantize_dequantize=_no_codec(name),
            bits_for_shape_fn=exl3_bits_for_shape(K),
        ))


_register()

PULSAR_QWEN_FORMATS = frozenset({f"EXL3_K{K}" for K in EXL3_KS} | {"MXFP8_E4M3", "BF16"})
