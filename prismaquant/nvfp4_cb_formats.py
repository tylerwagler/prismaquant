"""NVFP4-CB / gridbook codebook formats — independent encoder implementation.

Implements the encoder side of the gridbook codebook format family per the
normative spec (RobTand/gridbook docs/SPEC.md, Apache-2.0). The gridbook repo
ships the format spec and the vLLM serving plugin only; artifacts are produced
by the authors' non-public pipeline. This module is a clean-room encoder built
from the spec, API-compatible with the module gridbook's tests import
(``prismaquant.nvfp4_cb_formats``), so gridbook's own encode->decode round-trip
tests double as our conformance suite.

Scope (phase 1): FP4/E2M1 grid, ``product`` mode (n_sub=2), v1 (E4M3-direct)
and v2 (two-tier) scale codings, deterministic fixed-lattice codebooks, and
cost-stage quantize-dequantize entry points. FP8 grid / signed mode / full mode
raise ``NotImplementedError``.

Format math (spec §1):
  - 256-weight superblock along in_features; 32 codewords of 8 weights each.
  - codeword: k-bit index, product-split ``bit_split(k, 2)`` larger-part-first,
    sub0 in the low bits; each sub-index picks a 4-coord vector on the E2M1
    grid {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} from a shared sub-codebook.
  - v1 scales: 16 E4M3 bytes per superblock (one per group-16), NVFP4-style.
  - v2 scales: 1 E8M0 super byte + 16 4-bit codes into TWO_TIER_SUB_TABLE;
    composed scale T[c] * 2^(E-127) MUST be E4M3-exact in (0, 448].
  - effective bits: k/8 + 0.5 (v1), k/8 + 0.28125 (v2).
  - reconstruction: w = bf16_rn(codeword_value * scale)  (plugin numerics
    contract: decoded weights are bf16-rounded).
"""
from __future__ import annotations

import math

import torch

SUPERBLOCK = 256
VEC_DIM = 8
GROUP_SIZE = 16
N_SUB_FP4 = 2
SUB_DIM = VEC_DIM // N_SUB_FP4  # 4

SCALE_CODING_V1 = "e4m3_direct"
SCALE_CODING_TWO_TIER = "two_tier"

# E2M1 value grid (15 values).
E2M1_POS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_VALUES = tuple(sorted((-v for v in E2M1_POS)) + [0.0] + list(E2M1_POS))

# Default v2 sub-scale multiplier table (spec §1.2, T4_2oct8m): all 8 E4M3
# mantissa steps across 2 octaves. Every entry is E4M3-exact by construction.
TWO_TIER_SUB_TABLE = torch.tensor(
    [1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875,
     2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75], dtype=torch.float32)

_LATTICE_SEED = 0x9E3779B1  # deterministic; part of the encoder's identity
_LATTICE_SAMPLES = 262144   # 4-dim sub-vectors drawn for lattice construction
_LATTICE_ITERS = 30

_lattice_cache: dict[tuple[int, int], torch.Tensor] = {}
_two_tier_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _bit_split(k: int, n_sub: int) -> tuple[int, ...]:
    """Spec §1.1: as-even-as-possible split of k into n_sub parts, larger
    parts first (ceil-first). bit_split(13,2)=(7,6); bit_split(40,4)=(10,)*4."""
    base, rem = divmod(k, n_sub)
    return tuple(base + 1 if i < rem else base for i in range(n_sub))


def _e4m3_round(x: torch.Tensor) -> torch.Tensor:
    """Round-trip through float8_e4m3fn (RN), back to fp32."""
    return x.to(torch.float8_e4m3fn).to(torch.float32)


def nvfp4_cb_type_size(k: int, grid: str = "fp4",
                       scale_coding: str | None = None) -> int:
    """Bytes per 256-weight superblock (spec §1.4)."""
    if grid == "fp8":
        return 4 * k
    if grid != "fp4":
        raise ValueError(f"unknown grid {grid!r}")
    if scale_coding in (None, SCALE_CODING_V1):
        return 4 * k + 16
    if scale_coding == SCALE_CODING_TWO_TIER:
        return 4 * k + 9
    raise ValueError(f"unknown scale_coding {scale_coding!r}")


def nvfp4_cb_effective_bits(k: int, grid: str = "fp4",
                            scale_coding: str | None = None) -> float:
    return nvfp4_cb_type_size(k, grid, scale_coding) * 8.0 / SUPERBLOCK


def _two_tier_tables(device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(sub_table (16,), compose (256,16), legal (256,16) bool).

    compose[E, c] = T[c] * 2^(E-127); legal iff the product round-trips E4M3
    bit-exactly and lies in (0, 448] (spec §1.2)."""
    dev = torch.device(device)
    key = str(dev)
    if key in _two_tier_cache:
        return _two_tier_cache[key]
    T = TWO_TIER_SUB_TABLE.to(dev)
    E = torch.arange(256, dtype=torch.float32, device=dev) - 127.0
    compose = T.unsqueeze(0) * torch.exp2(E).unsqueeze(1)        # (256, 16)
    legal = (compose > 0) & (compose <= 448.0) \
        & torch.isfinite(compose) & (_e4m3_round(compose) == compose)
    out = (T, compose, legal)
    _two_tier_cache[key] = out
    return out


# --------------------------------------------------------------------------- #
# deterministic lattice codebooks
# --------------------------------------------------------------------------- #

def _snap_to_grid(x: torch.Tensor, nonneg: bool = False) -> torch.Tensor:
    """Per-coordinate nearest E2M1 value (half-grid incl. 0 when nonneg)."""
    vals = (0.0,) + E2M1_POS if nonneg else E2M1_VALUES
    grid = torch.tensor(vals, dtype=torch.float32, device=x.device)
    idx = (x.unsqueeze(-1) - grid).abs().argmin(dim=-1)
    return grid[idx]


def _lattice_sample(n: int, sub_dim: int) -> torch.Tensor:
    """Deterministic sample of group-normalized sub-vectors: draw Gaussian
    group-16 blocks, normalize by amax/6 (the ideal-scale convention), and
    slice into sub_dim chunks. Mirrors the distribution the encoder actually
    quantizes after scale normalization."""
    g = torch.Generator(device="cpu").manual_seed(_LATTICE_SEED)
    n_groups = (n * sub_dim + GROUP_SIZE - 1) // GROUP_SIZE
    blocks = torch.randn(n_groups, GROUP_SIZE, generator=g)
    amax = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    blocks = blocks * (6.0 / amax)
    return blocks.reshape(-1, sub_dim)[:n].contiguous()


def _dedupe_key(t: torch.Tensor) -> list[tuple]:
    return [tuple(row) for row in t.tolist()]


def _lattice_sub_table(bits: int, sub_dim: int = SUB_DIM) -> torch.Tensor:
    """Deterministic fixed lattice: 2^bits grid-valued sub_dim-vectors (fp16).

    Construction (fixed seed, fixed iteration count => bit-reproducible):
    seeded group-normalized Gaussian sample -> k-means++ init (seeded) ->
    Lloyd iterations with per-iteration snap to the E2M1 grid -> deterministic
    dedupe. Entry 0 is pinned to the all-zero vector."""
    key = (bits, sub_dim)
    if key in _lattice_cache:
        return _lattice_cache[key]
    B = 1 << bits
    x = _lattice_sample(_LATTICE_SAMPLES, sub_dim)

    # seeded k-means++ init
    g = torch.Generator(device="cpu").manual_seed(_LATTICE_SEED ^ bits)
    cent = torch.empty(B, sub_dim)
    cent[0] = 0.0
    d2 = (x - cent[0]).pow(2).sum(dim=1)
    for i in range(1, B):
        probs = d2.clamp_min(1e-12)
        pick = torch.multinomial(probs, 1, generator=g).item()
        cent[i] = x[pick]
        d2 = torch.minimum(d2, (x - cent[i]).pow(2).sum(dim=1))

    for _ in range(_LATTICE_ITERS):
        cent = _snap_to_grid(cent)
        cent[0] = 0.0
        assign = torch.cdist(x, cent).argmin(dim=1)
        sums = torch.zeros_like(cent).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=B).clamp_min(1)
        new = sums / counts.unsqueeze(1)
        empty = torch.bincount(assign, minlength=B) == 0
        new[empty] = cent[empty]        # keep empty clusters where they are
        new[0] = 0.0
        cent = new

    cent = _snap_to_grid(cent)
    cent[0] = 0.0

    # deterministic dedupe: replace duplicate rows with the sample points
    # worst-served by the current codebook (stable order), snapped.
    seen: set = set()
    dup_rows = []
    keys = _dedupe_key(cent)
    for i, k_ in enumerate(keys):
        if k_ in seen:
            dup_rows.append(i)
        else:
            seen.add(k_)
    if dup_rows:
        assign = torch.cdist(x, cent).argmin(dim=1)
        err = (x - cent[assign]).pow(2).sum(dim=1)
        order = torch.argsort(err, descending=True, stable=True)
        oi = 0
        for row in dup_rows:
            while oi < order.numel():
                cand = _snap_to_grid(x[order[oi]].unsqueeze(0))[0]
                oi += 1
                ck = tuple(cand.tolist())
                if ck not in seen:
                    cent[row] = cand
                    seen.add(ck)
                    break
    out = cent.to(torch.float16)
    _lattice_cache[key] = out
    return out


def learn_codebook(samples: torch.Tensor, k: int,
                   iters: int = _LATTICE_ITERS,
                   mode: str = "product") -> tuple[torch.Tensor, ...]:
    """Learn per-role product sub-codebooks from real scale-normalized
    sub-vector samples (spec §4: 'shared per-role learned codebook, pooled
    across layers and, for MoE, across experts'; ships in the sidecar).

    samples: (N, VEC_DIM) group-normalized weight vectors (w / group_scale).
    Deterministic: seeded k-means++ init, fixed Lloyd iterations, snap to the
    E2M1 grid each iteration, deterministic dedupe. Entry 0 pinned to zero.
    mode='signed': one non-negative 8-dim magnitude table (2^(k-8) entries)."""
    if mode == "signed":
        return (_kmeans_grid(samples.abs().to(torch.float32).contiguous(),
                             k - 8, seed=_LATTICE_SEED ^ (0x51 + k),
                             iters=iters, nonneg=True),)
    b0, b1 = _bit_split(k, N_SUB_FP4)
    subs = []
    for si, bits in enumerate((b0, b1)):
        lo = si * SUB_DIM
        x = samples[:, lo:lo + SUB_DIM].to(torch.float32).contiguous()
        subs.append(_kmeans_grid(x, bits, seed=_LATTICE_SEED ^ (bits + 977 * si),
                                 iters=iters))
    return tuple(subs)


def _kmeans_grid(x: torch.Tensor, bits: int, seed: int,
                 iters: int, nonneg: bool = False) -> torch.Tensor:
    """Seeded grid-snapped k-means over (N, sub_dim) samples -> fp16 table."""
    B = 1 << bits
    sub_dim = x.shape[1]
    n = x.shape[0]
    if n > _LATTICE_SAMPLES:
        g0 = torch.Generator(device="cpu").manual_seed(seed ^ 0x5bd1)
        x = x[torch.randperm(n, generator=g0)[:_LATTICE_SAMPLES]]
    g = torch.Generator(device="cpu").manual_seed(seed)
    cent = torch.empty(B, sub_dim)
    cent[0] = 0.0
    d2 = (x - cent[0]).pow(2).sum(dim=1)
    for i in range(1, B):
        pick = torch.multinomial(d2.clamp_min(1e-12), 1, generator=g).item()
        cent[i] = x[pick]
        d2 = torch.minimum(d2, (x - cent[i]).pow(2).sum(dim=1))
    for _ in range(iters):
        cent = _snap_to_grid(cent, nonneg)
        cent[0] = 0.0
        assign = torch.cdist(x, cent).argmin(dim=1)
        sums = torch.zeros_like(cent).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=B)
        new = sums / counts.clamp_min(1).unsqueeze(1)
        new[counts == 0] = cent[counts == 0]
        new[0] = 0.0
        cent = new
    cent = _snap_to_grid(cent, nonneg)
    cent[0] = 0.0
    seen: set = set()
    dup_rows = []
    for i, key in enumerate(_dedupe_key(cent)):
        if key in seen:
            dup_rows.append(i)
        else:
            seen.add(key)
    if dup_rows:
        assign = torch.cdist(x, cent).argmin(dim=1)
        err = (x - cent[assign]).pow(2).sum(dim=1)
        order = torch.argsort(err, descending=True, stable=True)
        oi = 0
        for row in dup_rows:
            while oi < order.numel():
                cand = _snap_to_grid(x[order[oi]].unsqueeze(0), nonneg)[0]
                oi += 1
                ck = tuple(cand.tolist())
                if ck not in seen:
                    cent[row] = cand
                    seen.add(ck)
                    break
    return cent.to(torch.float16)


def group_normalized_subvectors(w: torch.Tensor,
                                max_vectors: int = 1 << 20) -> torch.Tensor:
    """Extract (N, VEC_DIM) group-normalized vectors from a weight tensor,
    for learn_codebook. Uses the same amax/6 ideal-scale convention as the
    fast encoder's first pass."""
    w2 = w.reshape(-1, w.shape[-1]).to(torch.float32)
    rows, in_f = w2.shape
    g = w2.reshape(rows, in_f // GROUP_SIZE, GROUP_SIZE)
    amax = g.abs().amax(dim=-1, keepdim=True)
    q = torch.where(amax > 0, g * (6.0 / amax.clamp_min(1e-30)),
                    torch.zeros_like(g))
    v = q.reshape(-1, VEC_DIM)
    if v.shape[0] > max_vectors:
        gen = torch.Generator(device="cpu").manual_seed(_LATTICE_SEED ^ 0xA5A5)
        v = v[torch.randperm(v.shape[0], generator=gen)[:max_vectors]]
    return v.contiguous()


def _lattice_mag_table(bits: int) -> torch.Tensor:
    """Deterministic magnitude codebook for signed mode: 2^bits non-negative
    8-dim half-grid vectors, k-means on |group-normalized sample|."""
    key = (bits, -VEC_DIM)          # distinct cache namespace from product
    if key in _lattice_cache:
        return _lattice_cache[key]
    x = _lattice_sample(_LATTICE_SAMPLES, VEC_DIM).abs()
    out = _kmeans_grid(x, bits, seed=_LATTICE_SEED ^ (0x51 + bits),
                       iters=_LATTICE_ITERS, nonneg=True)
    _lattice_cache[key] = out
    return out


def _resolve_codebook(k: int, grid: str, mode: str, codebook, device):
    """Return the (tuple of) sub-codebook tensor(s) on `device`. `codebook`
    passthrough if given (tests pass explicit codebooks)."""
    if codebook is not None:
        if isinstance(codebook, (tuple, list)):
            return tuple(c.to(device) for c in codebook)
        return codebook.to(device)
    if grid == "fp4" and mode == "signed":
        if k <= 8:
            raise ValueError("signed mode needs k > 8")
        return (_lattice_mag_table(k - 8).to(device),)
    if grid != "fp4" or mode != "product":
        raise NotImplementedError(
            f"default codebook only for fp4 product/signed (got {grid}/{mode})")
    b0, b1 = _bit_split(k, N_SUB_FP4)
    return (_lattice_sub_table(b0).to(device), _lattice_sub_table(b1).to(device))


# --------------------------------------------------------------------------- #
# scale encoding
# --------------------------------------------------------------------------- #

def _encode_scales_v1(amax_g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(rows, n_sb, 16) group amax -> (scale_bytes uint8, scale fp32)."""
    ideal = amax_g / 6.0
    scale = _e4m3_round(ideal)
    # zero scale is legal in v1 (whole group reconstructs to 0)
    bytes_ = scale.to(torch.float8_e4m3fn).view(torch.uint8)
    return bytes_, scale


def _encode_scales_two_tier(amax_g: torch.Tensor, device
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(rows, n_sb, 16) group amax -> (super uint8 (rows,n_sb),
    sub_codes uint8 (rows,n_sb,16), scale fp32 (rows,n_sb,16)).

    Per spec §1.2: E covers the superblock's largest ideal scale; each group
    picks the nearest legal table entry, snapping UP when below the reachable
    set; all-zero groups take the first legal candidate; all-zero superblocks
    store the smallest legal E."""
    _, compose, legal = _two_tier_tables(device)
    e_any_legal = legal.any(dim=1)
    e_min_legal = int(torch.nonzero(e_any_legal)[0])
    e_max_legal = int(torch.nonzero(e_any_legal)[-1])

    ideal = amax_g / 6.0                                   # (rows, n_sb, 16)
    s_max = ideal.amax(dim=-1)                             # (rows, n_sb)
    zero_sb = s_max <= 0

    E = torch.floor(torch.log2(s_max.clamp_min(1e-38))) + 127.0
    E = E.clamp(e_min_legal, e_max_legal).to(torch.int64)
    E[zero_sb] = e_min_legal

    cand = compose[E]                                      # (rows, n_sb, 16)
    cand_legal = legal[E]
    # nearest legal entry per group (tie -> lowest index); illegal entries +inf
    diff = (cand.unsqueeze(-2) - ideal.unsqueeze(-1)).abs()  # (rows,n_sb,16g,16c)
    diff = diff.masked_fill(~cand_legal.unsqueeze(-2), float("inf"))
    c = diff.argmin(dim=-1).to(torch.uint8)                # (rows, n_sb, 16)

    # all-zero group -> first legal candidate at this E (deterministic)
    first_legal = cand_legal.float().argmax(dim=-1).to(torch.uint8)  # (rows,n_sb)
    zero_g = amax_g <= 0
    c = torch.where(zero_g, first_legal.unsqueeze(-1).expand_as(c), c)

    scale = torch.gather(cand, -1, c.long())               # (rows, n_sb, 16)
    return E.to(torch.uint8), c, scale


# --------------------------------------------------------------------------- #
# encode / fields
# --------------------------------------------------------------------------- #

def _assign_product(qvals: torch.Tensor, subs: tuple[torch.Tensor, ...],
                    col_weights: torch.Tensor | None,
                    chunk: int = 1 << 18) -> list[torch.Tensor]:
    """qvals (rows, in) scale-normalized; assign each sub_dim slice to its
    nearest sub-codebook entry (weighted L2 if col_weights given).
    Returns per-sub index tensors (rows, in/VEC_DIM)."""
    rows, in_f = qvals.shape
    n_vec = in_f // VEC_DIM
    out = []
    for si, cb in enumerate(subs):
        cbf = cb.to(torch.float32)                          # (B, sub_dim)
        lo = si * SUB_DIM
        v = qvals.reshape(rows, n_vec, VEC_DIM)[:, :, lo:lo + SUB_DIM]
        v = v.reshape(-1, SUB_DIM)                          # (N, sub_dim)
        if col_weights is not None:
            wcol = col_weights.reshape(n_vec, VEC_DIM)[:, lo:lo + SUB_DIM]
            wcol = wcol.to(torch.float32).clamp_min(1e-12)
            # per-vector weights vary by column position only
            wv = wcol.unsqueeze(0).expand(rows, n_vec, SUB_DIM).reshape(-1, SUB_DIM)
        else:
            wv = None
        idx = torch.empty(v.shape[0], dtype=torch.int32, device=v.device)
        for s in range(0, v.shape[0], chunk):
            ve = v[s:s + chunk]
            if wv is None:
                d = torch.cdist(ve, cbf).pow_(2)
            else:
                we = wv[s:s + chunk]
                # sum_j w_j (x_j - c_j)^2 = (w x^2)·1 - 2 (w x)·c + w·c^2
                d = ((we * ve * ve).sum(-1, keepdim=True)
                     - 2.0 * (we * ve) @ cbf.T
                     + we @ (cbf * cbf).T)
            idx[s:s + chunk] = d.argmin(dim=1).to(torch.int32)
        out.append(idx.reshape(rows, n_vec))
    return out


def _decode_codes(codes: torch.Tensor, subs, k: int,
                  mode: str = "product") -> torch.Tensor:
    """(rows, n_vec) codewords -> (rows, in) codeword values (fp32)."""
    if mode == "signed":
        mag = subs[0].to(torch.float32)[(codes >> 8).long()]  # (rows,n_vec,8)
        sign_bits = ((codes.unsqueeze(-1) >> torch.arange(
            VEC_DIM, device=codes.device)) & 1)
        cw = mag * torch.where(sign_bits.bool(), -1.0, 1.0)
        return cw.reshape(codes.shape[0], -1)
    b0, b1 = _bit_split(k, N_SUB_FP4)
    i0 = (codes & ((1 << b0) - 1)).long()
    i1 = ((codes >> b0) & ((1 << b1) - 1)).long()
    cw = torch.cat([subs[0].to(torch.float32)[i0],
                    subs[1].to(torch.float32)[i1]], dim=-1)
    return cw.reshape(codes.shape[0], -1)


def _assign_signed(qvals: torch.Tensor, mag_cb: torch.Tensor,
                   col_weights: torch.Tensor | None,
                   chunk: int = 1 << 17) -> torch.Tensor:
    """Signed-mode assignment: codeword = 8 sign bits (bit j = coord j
    negative) | magnitude index << 8, magnitude = nearest codebook entry to
    |q| (weighted L2). Returns (rows, n_vec) int64 codes."""
    rows, in_f = qvals.shape
    n_vec = in_f // VEC_DIM
    v = qvals.reshape(-1, VEC_DIM)
    cbf = mag_cb.to(torch.float32)
    m = v.abs()
    if col_weights is not None:
        wcol = col_weights.reshape(n_vec, VEC_DIM).to(torch.float32) \
            .clamp_min(1e-12)
        wv = wcol.unsqueeze(0).expand(rows, n_vec, VEC_DIM).reshape(-1, VEC_DIM)
    else:
        wv = None
    idx = torch.empty(m.shape[0], dtype=torch.int64, device=m.device)
    for s in range(0, m.shape[0], chunk):
        me = m[s:s + chunk]
        if wv is None:
            d = torch.cdist(me, cbf).pow_(2)
        else:
            we = wv[s:s + chunk]
            d = ((we * me * me).sum(-1, keepdim=True)
                 - 2.0 * (we * me) @ cbf.T + we @ (cbf * cbf).T)
        idx[s:s + chunk] = d.argmin(dim=1)
    signs = (v < 0).to(torch.int64)
    bitw = (1 << torch.arange(VEC_DIM, device=v.device, dtype=torch.int64))
    sign_word = (signs * bitw).sum(dim=-1)
    return (sign_word | (idx << 8)).reshape(rows, n_vec)


def _refit_scales(w2: torch.Tensor, cvals: torch.Tensor, scale: torch.Tensor,
                  scale_coding: str, super_e, device, prev_sub_codes=None):
    """Closed-form per-group scale refit given fixed codeword values:
    s* = <w, c> / <c, c> per group-16, snapped to the legal scale set
    (v1: E4M3 grid; v2: legal compose entries at the superblock's E).
    Returns (scale, scale_bytes, sub_codes) with the non-relevant ones None."""
    rows, in_f = w2.shape
    n_sb = in_f // SUPERBLOCK
    wg = w2.reshape(rows, n_sb, 16, GROUP_SIZE)
    cg = cvals.reshape(rows, n_sb, 16, GROUP_SIZE)
    num = (wg * cg).sum(dim=-1)
    den = (cg * cg).sum(dim=-1)
    s_star = torch.where(den > 0, num / den.clamp_min(1e-30),
                         torch.zeros_like(num)).clamp_min(0.0)
    if scale_coding == SCALE_CODING_V1:
        new_scale = _e4m3_round(s_star)
        # keep old scale where refit collapses to zero but group had signal
        new_scale = torch.where((new_scale > 0) | (scale <= 0),
                                new_scale, scale)
        bytes_ = new_scale.to(torch.float8_e4m3fn).view(torch.uint8)
        return new_scale, bytes_, None
    _, compose, legal = _two_tier_tables(device)
    cand = compose[super_e.long()]                        # (rows, n_sb, 16c)
    cand_legal = legal[super_e.long()]
    diff = (cand.unsqueeze(-2) - s_star.unsqueeze(-1)).abs()
    diff = diff.masked_fill(~cand_legal.unsqueeze(-2), float("inf"))
    c = diff.argmin(dim=-1).to(torch.uint8)
    new_scale = torch.gather(cand, -1, c.long())
    # zero-signal groups keep their previous deterministic (scale, code) pair
    keep = s_star <= 0
    if prev_sub_codes is not None:
        c = torch.where(keep, prev_sub_codes, c)
    return torch.where(keep, scale, new_scale), None, c


def nvfp4_cb_fields(w: torch.Tensor, k: int, grid: str = "fp4",
                    mode: str = "product", codebook=None,
                    scale_coding: str | None = None,
                    encode_tier: str = "fast",
                    col_weights: torch.Tensor | None = None) -> dict:
    """Encode weights to CB fields (numerics only; bytes via assemble).

    w: (out, in) or (E, out, in), any float dtype. Returns a dict with the
    index/scale fields plus enough meta to reconstruct or pack.

    encode_tier: 'fast' = amax scales + one assignment pass.
                 'em'   = fast, then alternate closed-form scale refit /
                          re-assignment (2 rounds). Better, ~3x slower."""
    if grid != "fp4" or mode not in ("product", "signed"):
        raise NotImplementedError("phase 1: fp4 product/signed only")
    if scale_coding is None:
        scale_coding = SCALE_CODING_V1
    orig_shape = tuple(w.shape)
    w2 = w.reshape(-1, orig_shape[-1]).to(torch.float32)
    rows, in_f = w2.shape
    if in_f % SUPERBLOCK:
        raise ValueError(f"in_features {in_f} not a multiple of {SUPERBLOCK}")
    n_sb = in_f // SUPERBLOCK
    device = w2.device
    subs = _resolve_codebook(k, grid, mode, codebook, device)

    amax_g = w2.reshape(rows, n_sb, SUPERBLOCK // GROUP_SIZE, GROUP_SIZE) \
               .abs().amax(dim=-1)                          # (rows, n_sb, 16)
    if scale_coding == SCALE_CODING_V1:
        scale_bytes, scale = _encode_scales_v1(amax_g)
        super_e, sub_codes = None, None
    else:
        super_e, sub_codes, scale = _encode_scales_two_tier(amax_g, device)
        scale_bytes = None

    b_widths = _bit_split(k, N_SUB_FP4)

    def assign(cur_scale):
        scale_per_w = cur_scale.unsqueeze(-1) \
            .expand(rows, n_sb, 16, GROUP_SIZE).reshape(rows, in_f)
        qvals = torch.where(scale_per_w > 0,
                            w2 / scale_per_w.clamp_min(1e-38),
                            torch.zeros_like(w2))
        if mode == "signed":
            return _assign_signed(qvals, subs[0], col_weights)
        sub_idx = _assign_product(qvals, subs, col_weights)
        return sub_idx[0].to(torch.int64) \
            | (sub_idx[1].to(torch.int64) << b_widths[0])

    codes = assign(scale)

    if encode_tier == "em":
        for _ in range(2):
            cvals = _decode_codes(codes, subs, k, mode)
            scale, nb, nc = _refit_scales(w2, cvals, scale, scale_coding,
                                          super_e, device,
                                          prev_sub_codes=sub_codes)
            if scale_coding == SCALE_CODING_V1:
                scale_bytes = nb
            else:
                sub_codes = nc
            codes = assign(scale)

    return {
        "codes": codes,                    # (rows, in/8) int64 k-bit codewords
        "scale_bytes": scale_bytes,        # v1: (rows, n_sb, 16) uint8 | None
        "super": super_e,                  # v2: (rows, n_sb) uint8 | None
        "sub_codes": sub_codes,            # v2: (rows, n_sb, 16) uint8 | None
        "scale": scale,                    # (rows, n_sb, 16) fp32 decoded
        "codebook": subs,
        "k": k, "grid": grid, "mode": mode,
        "scale_coding": scale_coding,
        "shape": orig_shape,
    }


# --------------------------------------------------------------------------- #
# reconstruct / pack / unpack
# --------------------------------------------------------------------------- #

def nvfp4_cb_reconstruct(fields: dict, out_dtype=torch.bfloat16) -> torch.Tensor:
    """fields -> dense weights, w = bf16_rn(codeword_value * scale) per the
    plugin numerics contract; returned in out_dtype."""
    codes = fields["codes"]
    subs = fields["codebook"]
    k = fields["k"]
    rows, n_vec = codes.shape
    in_f = n_vec * VEC_DIM
    cw = _decode_codes(codes, subs, k, fields.get("mode", "product"))
    scale = fields["scale"]                                  # (rows,n_sb,16)
    scale_per_w = scale.unsqueeze(-1) \
        .expand(rows, scale.shape[1], 16, GROUP_SIZE) \
        .reshape(rows, in_f)
    w = cw * scale_per_w
    w = w.to(torch.bfloat16).to(out_dtype)
    return w.reshape(fields["shape"])


def nvfp4_cb_assemble_bytes(fields: dict, k: int, grid: str = "fp4",
                            mode: str = "product") -> torch.Tensor:
    """fields -> on-disk uint8 stream (rows, n_sb * type_size), spec §1:
    per superblock [4k bytes LSB-first index stream][scale plane]."""
    codes = fields["codes"]
    rows, n_vec = codes.shape
    n_sb = n_vec * VEC_DIM // SUPERBLOCK
    ts = nvfp4_cb_type_size(k, grid, fields["scale_coding"])
    sb_codes = codes.reshape(rows * n_sb, 32)

    # LSB-first bit packing: 32 k-bit codewords -> 4k bytes
    bit_idx = torch.arange(k, device=codes.device)
    bits = ((sb_codes.unsqueeze(-1) >> bit_idx) & 1).to(torch.uint8)
    bits = bits.reshape(rows * n_sb, 32 * k)               # LSB-first stream
    byte_w = (1 << torch.arange(8, device=codes.device)).to(torch.uint8)
    idx_bytes = (bits.reshape(rows * n_sb, 4 * k, 8) * byte_w).sum(
        dim=-1, dtype=torch.int64).to(torch.uint8)

    if fields["scale_coding"] == SCALE_CODING_V1:
        plane = fields["scale_bytes"].reshape(rows * n_sb, 16)
    else:
        sub = fields["sub_codes"].reshape(rows * n_sb, 16).to(torch.int64)
        packed = (sub[:, 0::2] | (sub[:, 1::2] << 4)).to(torch.uint8)  # even=low
        plane = torch.cat([fields["super"].reshape(rows * n_sb, 1), packed],
                          dim=1)
    out = torch.cat([idx_bytes, plane], dim=1)             # (rows*n_sb, ts)
    assert out.shape[1] == ts
    return out.reshape(rows, n_sb * ts).contiguous()


def nvfp4_cb_unpack(qweight: torch.Tensor, k: int, grid: str = "fp4",
                    mode: str = "product",
                    scale_coding: str | None = None,
                    codebook=None) -> dict:
    """on-disk bytes (rows, n_sb*type_size) -> fields (inverse of assemble)."""
    if scale_coding is None:
        scale_coding = SCALE_CODING_V1
    ts = nvfp4_cb_type_size(k, grid, scale_coding)
    rows = qweight.shape[0]
    n_sb = qweight.shape[1] // ts
    sb = qweight.reshape(rows * n_sb, ts)
    idx_bytes = sb[:, :4 * k].to(torch.int64)

    bit_w = torch.arange(8, device=qweight.device)
    bits = ((idx_bytes.unsqueeze(-1) >> bit_w) & 1)
    bits = bits.reshape(rows * n_sb, 32 * k).reshape(rows * n_sb, 32, k)
    kw = (1 << torch.arange(k, device=qweight.device, dtype=torch.int64))
    codes = (bits * kw).sum(dim=-1).reshape(rows, n_sb * 32)

    device = qweight.device
    subs = _resolve_codebook(k, grid, mode, codebook, device)
    fields = {
        "codes": codes, "codebook": subs, "k": k, "grid": grid, "mode": mode,
        "scale_coding": scale_coding,
        "shape": (rows, n_sb * SUPERBLOCK),
        "scale_bytes": None, "super": None, "sub_codes": None,
    }
    if scale_coding == SCALE_CODING_V1:
        plane = sb[:, 4 * k:].reshape(rows, n_sb, 16)
        fields["scale_bytes"] = plane
        fields["scale"] = plane.view(torch.float8_e4m3fn).to(torch.float32)
    else:
        super_e = sb[:, 4 * k].reshape(rows, n_sb)
        packed = sb[:, 4 * k + 1:].reshape(rows, n_sb, 8).to(torch.int64)
        sub = torch.empty(rows, n_sb, 16, dtype=torch.uint8, device=device)
        sub[..., 0::2] = (packed & 0xF).to(torch.uint8)
        sub[..., 1::2] = (packed >> 4).to(torch.uint8)
        _, compose, _ = _two_tier_tables(device)
        fields["super"] = super_e
        fields["sub_codes"] = sub
        fields["scale"] = compose[super_e.long().unsqueeze(-1), sub.long()]
    return fields


# --------------------------------------------------------------------------- #
# cost-stage entry points
# --------------------------------------------------------------------------- #

def parse_cb_format_name(name: str) -> tuple[int, str, str]:
    """'NVFP4_CB_K16' -> (16, 'two_tier', 'product');
    'NVFP4_CB_S16' -> (16, 'two_tier', 'signed'); trailing 'V1' selects v1."""
    for prefix, mode in (("NVFP4_CB_K", "product"), ("NVFP4_CB_S", "signed")):
        if name.startswith(prefix):
            rest = name[len(prefix):]
            if rest.endswith("V1"):
                return int(rest[:-2]), SCALE_CODING_V1, mode
            return int(rest), SCALE_CODING_TWO_TIER, mode
    raise ValueError(name)


def nvfp4_cb_quantize_dequantize(w: torch.Tensor, fmt_name: str,
                                 col_weights: torch.Tensor | None = None,
                                 encode_tier: str = "fast",
                                 codebook=None) -> torch.Tensor:
    """RTN-style weight QDQ for the cost stage: encode -> reconstruct."""
    k, coding, mode = parse_cb_format_name(fmt_name)
    fields = nvfp4_cb_fields(w, k, mode=mode, scale_coding=coding,
                             codebook=codebook, encode_tier=encode_tier,
                             col_weights=col_weights)
    return nvfp4_cb_reconstruct(fields, out_dtype=w.dtype)


def fp4_group16_act_qdq(x: torch.Tensor) -> torch.Tensor:
    """W4A4 activation QDQ, bit-matched to gridbook codec.fp4_group16_act_qdq:
    dynamic group-16 scale = amax.clamp_min(1e-8)/6 (raw fp32, NOT e4m3-
    rounded), values RTN to E2M1 (ties toward the lower grid value)."""
    orig = x.shape
    xf = x.reshape(-1, orig[-1]).to(torch.float32)
    rows, cols = xf.shape
    pad = (-cols) % GROUP_SIZE
    if pad:
        xf = torch.nn.functional.pad(xf, (0, pad))
    g = xf.reshape(rows, -1, GROUP_SIZE)
    scale = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 6.0
    q = _snap_to_grid(g / scale) * scale
    q = q.reshape(rows, cols + pad)[:, :cols]
    return q.to(x.dtype).reshape(orig)
