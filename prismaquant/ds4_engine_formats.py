"""What the pulsar ds4 engine's artifact writer can emit -- the `ds4_engine` serving lane's exporter declaration.

The exporter for this lane is pulsar's `gguf-tools/deepseek4-quantize` (C, out of tree; driven by a per-tensor format map).  A
serving profile's export lane must point at *the exporter's own* format table (`serving_profiles.ExportLaneSpec`), and that table
cannot live in prismaquant for an out-of-tree exporter, so this module is the declaration the lane is bound to.  Keep it equal to
the quantizer's format map vocabulary (`gguf-tools/prisma/*format-map*.json`): a name here that the quantizer cannot emit would let
the allocator spend budget on a rung that fails at build.

Kernel inventory as of 2026-09 (pulsar dev): routed experts IQ2_XXS (k-major MMQ, artifact type 44), Q2_K, and the checkpoint-native
MXFP4 grid (CUTLASS_MXFP4, type 40); attention / shared-expert / dense projections on the FP8 tensor-core path (MXFP8_E4M3 group-32,
MXFP8_LT type 41); embedding and output head as plain bf16 (the bf16 head is a deliberate fidelity choice, 3.5% decode paid).
"""

DS4_ENGINE_FORMATS = frozenset({
    "IQ2_XXS",
    "Q2_K",
    "MXFP4",
    "CUTLASS_MXFP4",
    "MXFP8_E4M3",
    "BF16",
})
