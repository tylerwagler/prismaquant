"""The on-disk precompute cache round-trips for a multi-stream residual and
comes back memory-mapped with host-staged activations.

DSv4's hidden state is `[B, S, hc_mult, H]`; the geometry check accepted only
`[B, S, H]`, so every DSv4 run refused its own cache and recomputed phase 1
(and the reload after a fresh compute would have refused too)."""
from __future__ import annotations

import torch

from prismaquant.incremental_probe import (
    _HostStagedActivations,
    _PRECOMPUTE_CACHE_KEYS,
    _PRECOMPUTE_CACHE_SCHEMA,
    _load_precompute_cache,
)


def _payload(meta: dict, act_shape: tuple[int, ...], n_layers: int = 3) -> dict:
    ids = torch.zeros(act_shape[0], act_shape[1], dtype=torch.int64)
    acts = [torch.randn(*act_shape, dtype=torch.bfloat16) for _ in range(n_layers + 1)]
    data = {key: {} for key in _PRECOMPUTE_CACHE_KEYS}
    data.update({
        "schema": _PRECOMPUTE_CACHE_SCHEMA,
        "activations_cpu": acts,
        "grad_at_tail": torch.randn(*act_shape, dtype=torch.bfloat16),
        "ids_cpu": ids,
        "shared_pass_state": None,
        "meta": dict(meta),
    })
    return data


def test_multistream_cache_loads_memory_mapped_and_host_staged(tmp_path):
    meta = {"fingerprint": "t"}
    path = tmp_path / "precomputed.pt"
    torch.save(_payload(meta, (2, 5, 4, 8)), path)
    pre = _load_precompute_cache(path, meta, torch.device("cpu"))
    assert pre is not None
    assert isinstance(pre.activations_cpu, _HostStagedActivations)
    assert len(pre.activations_cpu) == 4
    a = pre.activations_cpu[1]
    assert tuple(a.shape) == (2, 5, 4, 8)
    # a read is an anonymous copy, never the file-backed view
    raw = list.__getitem__(pre.activations_cpu, 1)
    assert a.data_ptr() != raw.data_ptr()
    assert torch.equal(a, raw)
    assert tuple(pre.grad_at_tail.shape) == (2, 5, 4, 8)


def test_single_stream_cache_still_loads(tmp_path):
    meta = {"fingerprint": "t"}
    path = tmp_path / "precomputed.pt"
    torch.save(_payload(meta, (2, 5, 8)), path)
    assert _load_precompute_cache(path, meta, torch.device("cpu")) is not None


def test_wrong_rank_is_refused(tmp_path):
    meta = {"fingerprint": "t"}
    path = tmp_path / "precomputed.pt"
    torch.save(_payload(meta, (2, 5, 2, 2, 8)), path)
    assert _load_precompute_cache(path, meta, torch.device("cpu")) is None
