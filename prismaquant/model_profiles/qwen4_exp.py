"""Qwen4Exp (``model_type: qwen4_exp``) — Qwen3.8-Flash-Next family profile.

Scaffolding for the 177B checkpoint at ``/mnt/shared/models/Qwen3.8-Flash-Next``
(48 text layers, 512 routed experts + 1 shared expert, gated-DeltaNet linear
attention with a full-attention layer every 4th position, a QSA sparse-attention
indexer inside each full-attention layer, hyper-connections with ``hc_count=4``
residual streams, one PLE n-gram-embedding layer, an MTP sidecar, and a vision
tower).

Every naming claim below is traceable to one of two sources, cited inline:

  * ``transformers/models/qwen4_exp/modeling_qwen4_exp.py`` (transformers
    5.16.1, in ``/home/rob/dq-runs/venvs/prismaquant-tf516``) — the module and
    parameter names the live model exposes;
  * ``/mnt/shared/models/Qwen3.8-Flash-Next/model.safetensors.index.json`` —
    the on-disk source keys.

**There is no vLLM class for this architecture in this environment** (``import
vllm`` raises ``ModuleNotFoundError`` in the tf516 venv, and no vLLM release
ships a ``Qwen4Exp*`` model). Tier-1 auto-derivation via
``packed_modules_mapping``/``hf_to_vllm_mapper`` (``base.py`` ``_ensure_vllm_class``)
is therefore *impossible*, not merely unused: ``vllm_architecture_class()``
returns ``None`` and every structural fact is declared by hand in
``specs/qwen4_exp.json``. Consequently the spec deliberately declares **no**
``recipe_to_vllm`` naming map, no ``moe.per_expert_regex``, no
``default_serving_profile`` and no ``supported_lanes``: each of those is a
statement about what a serving runtime does, and CLAUDE.md principle 14 says
such a statement is attested or refused — never asserted from the producer
side. They are owed once a vLLM (or Gridbook) class exists to attest them.

This profile is **not yet registered** in ``registry.py`` (the scaffolding task
was scoped to new files only). Until the registration snippet in
``/home/rob/dq-runs/coordination/qwen4exp-structure-recon-2026-08-26.md`` is
applied, ``detect_profile()`` resolves this architecture to a ``SpecMatchProfile``
built from ``specs/qwen4_exp.json`` — which carries the naming/fusion/packing
declarations but *not* the Python-side walk rules or the RMSNorm offset below.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .base import ModelProfile

#: Checkpoint-key marker of the PLE n-gram table (128 row-contiguous shards,
#: 51.2B bf16 params, ~102 GB). transformers' own ``_no_placement_params``
#: keeps it off the device; the streaming loader must never read it either.
NGRAM_SHARD_MARKER = ".ple_embedding.ngram_embedding.shard_"

#: Checkpoint directories declared to a qwen4_exp profile whose index carries
#: the n-gram shards; the disk table reads from the first one.
_NGRAM_SOURCES: list[Path] = []


class _DeviceTag:
    """Stands in for ``ngram_embedding.weight`` where HF only reads ``.device``."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Qwen4ExpDiskNGramTable(nn.Module):
    """The PLE n-gram table as a parameter-free, disk-backed row gather.

    ``Qwen4ExpTextNGramEmbedding`` concatenates ``shard_0 .. shard_{S-1}``
    (each ``[rows, 160]`` bf16) into one ``nn.Embedding`` (transformers'
    conversion mapping, ``Concatenate(dim=0)``) and then only ever looks rows
    up. This module performs the same lookup straight from the safetensors
    files through a read-only numpy memmap, so only the touched rows are paged
    in and nothing of the 102 GB table is materialised on the host or device.
    It carries no parameter or buffer: the table is a lookup, not a weight the
    probe prices or the allocator places (it ships on the disk path).
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, layer_idx: int):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.layer_idx = int(layer_idx)
        self._shards = None
        self._rows_per_shard = None

    @property
    def weight(self):
        return _DeviceTag()

    def _open(self):
        if self._shards is not None:
            return
        if not _NGRAM_SOURCES:
            raise RuntimeError(
                "qwen4_exp n-gram disk table: no checkpoint with n-gram shards "
                "was declared to the profile (detect_profile(model_path))")
        src = _NGRAM_SOURCES[0]
        weight_map = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
        prefix = f"model.language_model.layers.{self.layer_idx}.ple.ple_embedding.ngram_embedding.shard_"
        keys = sorted((k for k in weight_map if k.startswith(prefix)),
                      key=lambda k: int(k[len(prefix):].split(".")[0]))
        if not keys:
            raise RuntimeError(f"qwen4_exp n-gram disk table: no {prefix}* keys in {src}")
        shards = []
        for i, key in enumerate(keys):
            if int(key[len(prefix):].split(".")[0]) != i:
                raise RuntimeError(f"n-gram shard numbering has a gap at {key}")
            path = src / weight_map[key]
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            meta = header[key]
            if meta["dtype"] != "BF16" or meta["shape"][1] != self.embedding_dim:
                raise RuntimeError(f"n-gram shard {key}: {meta['dtype']} {meta['shape']}")
            start, end = meta["data_offsets"]
            rows = meta["shape"][0]
            if end - start != rows * self.embedding_dim * 2:
                raise RuntimeError(f"n-gram shard {key}: byte span {end - start} != shape")
            shards.append(np.memmap(path, dtype=np.uint16, mode="r", offset=8 + n + start,
                                    shape=(rows, self.embedding_dim)))
        rows = {s.shape[0] for s in shards}
        if len(rows) != 1:
            raise RuntimeError(f"n-gram shards are not equal-sized: {sorted(rows)}")
        self._rows_per_shard = rows.pop()
        if self._rows_per_shard * len(shards) < self.num_embeddings - self._rows_per_shard:
            raise RuntimeError("n-gram shards cover fewer rows than the embedding declares")
        self._shards = shards

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        self._open()
        flat = ids.reshape(-1).to("cpu")
        uniq, inv = torch.unique(flat, return_inverse=True)
        u = uniq.numpy()
        shard_of = u // self._rows_per_shard
        if u.size and (u.min() < 0 or shard_of.max() >= len(self._shards)):
            raise IndexError("n-gram id outside the table")
        out = np.empty((u.size, self.embedding_dim), dtype=np.uint16)
        for s in np.unique(shard_of):
            sel = np.nonzero(shard_of == s)[0]
            out[sel] = self._shards[int(s)][u[sel] - int(s) * self._rows_per_shard]
        rows = torch.from_numpy(out).view(torch.bfloat16).to(ids.device)
        return rows[inv.to(ids.device)].view(*ids.shape, self.embedding_dim)


def _install_disk_ngram_table() -> None:
    """Patch ``Qwen4ExpTextNGramEmbedding`` so every skeleton built from here
    on carries the disk table instead of a 51B-parameter ``nn.Embedding``."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    cls = modeling.Qwen4ExpTextNGramEmbedding
    if getattr(cls, "_prismaquant_disk_table", False):
        return
    original_init = cls.__init__

    def __init__(self, config, embedding_dim, layer_idx, ple_layer_index=0):
        original_init(self, config, embedding_dim, layer_idx, ple_layer_index)
        emb = self.ngram_embedding
        self.ngram_embedding = Qwen4ExpDiskNGramTable(
            emb.num_embeddings, emb.embedding_dim, layer_idx)

    cls.__init__ = __init__
    cls._prismaquant_disk_table = True

    # The QSA indexer gathers per-sequence rope rows (`full_cos[batch_idx]`),
    # so it needs position embeddings with the real batch dimension, which
    # `Qwen4ExpTextModel.forward` gets by expanding position_ids to
    # (4, B, T). The streamed driver computes one broadcastable row (batch 1);
    # expanding it here is the identical tensor the model forward builds.
    indexer = modeling.Qwen4ExpTextQSAIndexer
    original_forward = indexer.forward

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
        cos, sin = position_embeddings
        batch = hidden_states.shape[0]
        if cos.shape[0] == 1 and batch > 1:
            position_embeddings = (cos.expand(batch, *cos.shape[1:]),
                                   sin.expand(batch, *sin.shape[1:]))
        return original_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values)

    indexer.forward = forward


class Qwen4ExpProfile(ModelProfile):
    """Model profile for the ``qwen4_exp`` (Qwen3.8-Flash-Next) family."""

    # Detection priority (lower = consulted first). 200 sits after every
    # currently-registered profile (Laguna is 190); nothing else claims
    # `qwen4_exp`, so the ordering is only a tie-break placeholder. It MUST
    # equal `specs/qwen4_exp.json`'s `priority`
    # (tests/test_spec_match_profile.py asserts the two agree).
    priority = 200

    @property
    def name(self) -> str:
        return "qwen4_exp"

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        """Claim the multimodal wrapper, the text carve-out, and the inner
        text config.

        `config.json` of the 177B checkpoint declares
        ``model_type: "qwen4_exp"`` with ``architectures:
        ["Qwen4ExpForConditionalGeneration"]``; its ``text_config`` declares
        ``model_type: "qwen4_exp_text"``. A staged text-only checkpoint
        promotes the inner type (see `staging.promote_inner_model_type` in the
        spec), so both must be claimed.
        """
        mt = (model_type or "").lower()
        if mt in {"qwen4_exp", "qwen4_exp_text"}:
            return True
        return any(str(a).startswith("Qwen4Exp") for a in (architectures or []))

    # ------------------------------------------------------------
    # vLLM: absent by measurement, not by omission
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # Streaming probe glue (Qwen4ExpTextModel.forward, read 2026-09-26 from
    # transformers 5.18.0.dev0 modeling_qwen4_exp.py)
    # ------------------------------------------------------------
    def _declare_model_path(self, model_path) -> None:
        super()._declare_model_path(model_path)
        index = Path(model_path) / "model.safetensors.index.json"
        if index.is_file() and Path(model_path) not in _NGRAM_SOURCES:
            if any(NGRAM_SHARD_MARKER in k for k in json.loads(index.read_text())["weight_map"]):
                _NGRAM_SOURCES.append(Path(model_path))

    def register_vendored_modeling(self) -> None:
        """Replace the 51B-parameter n-gram ``nn.Embedding`` with the disk table."""
        _install_disk_ngram_table()

    def checkpoint_to_live_name(self, ckpt_key: str, *, multimodal: bool = False):
        """Drop the n-gram table shards: the disk table reads them itself."""
        if NGRAM_SHARD_MARKER in ckpt_key:
            return None
        return super().checkpoint_to_live_name(ckpt_key, multimodal=multimodal)

    def rotary_position_ids(self, position_ids):
        """Three identical mRoPE rows for text, as ``Qwen4ExpTextModel.forward``
        passes ``position_ids[1:]`` of its 4-row text layout to the rotary."""
        if position_ids.ndim == 2:
            return position_ids.unsqueeze(0).expand(3, -1, -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 3:
            return position_ids
        raise ValueError("qwen4_exp rotary positions require [batch, tokens] or [3, batch, tokens]")

    def expand_hidden_for_layers(self, hidden, base_model):
        """``hidden_states.repeat(1, 1, hc_count)``: 4 identical residual streams."""
        return hidden.repeat(1, 1, base_model.config.hc_count)

    def collapse_hidden_after_layers(self, hidden, base_model):
        """``self.hyper_connection_mixer(hidden_states)``: the top-level gated
        read collapses the streams; its grouped hc_norm is the final norm."""
        return base_model.hyper_connection_mixer(hidden)

    def final_norm(self, base_model):
        """Identity: the mixer's hc_norm (inside `collapse_hidden_after_layers`)
        is this family's only pre-head norm; `Qwen4ExpTextModel` has no `norm`."""
        if hasattr(base_model, "norm"):
            raise RuntimeError("qwen4_exp text model grew a `norm`; revisit final_norm")
        return nn.Identity()

    def head_resident_extra_prefixes(self, root) -> list[str]:
        """The top-level stream mixer (``model.hyper_connection_mixer``) runs in
        `collapse_hidden_after_layers`, so it loads with the head batch."""
        if root is not None and not hasattr(root, "model") and hasattr(root, "hyper_connection_mixer"):
            return ["hyper_connection_mixer."]
        return ["model.hyper_connection_mixer."]

    def extra_layer_kwargs(self, *, input_ids=None) -> dict:
        """The PLE layer hashes the token ids (``ple_input_ids``)."""
        return {"ple_input_ids": input_ids} if input_ids is not None else {}

    def vllm_architecture_class(self) -> str | None:
        """None — no vLLM class exists for ``Qwen4Exp*``.

        Returning a name here would make `base.py`'s auto-derivation import a
        class that does not exist and, worse, would encode an unattested claim
        about vLLM's fused/packed mapping. The fused-sibling groups and packed
        expert layout are declared manually in the spec instead, read from HF
        modelling code.
        """
        return None

    # ------------------------------------------------------------
    # Numerics
    # ------------------------------------------------------------
    def rms_norm_parameter_offset(self) -> float | None:
        """1.0 — this family executes ``gamma = 1 + weight``.

        `Qwen4ExpTextRMSNorm.__init__` initialises ``self.weight`` to *zeros*
        (modeling_qwen4_exp.py:162) and its forward computes
        ``output * (1.0 + self.weight.float())`` (modeling_qwen4_exp.py:177),
        exactly like Qwen3.5/3.6.

        Caveat for any future function-preserving transform: the *gated*
        variant used inside the DeltaNet block, `Qwen4ExpTextRMSNormGated`,
        is a different encoding — ones-init (modeling_qwen4_exp.py:187) and a
        plain ``self.weight * hidden_states`` (modeling_qwen4_exp.py:196), i.e.
        offset 0.0. This method returns the family-wide value for the plain
        norm; a transform that touches ``linear_attn.norm`` must special-case
        it rather than inherit this answer.
        """
        return 1.0

    # ------------------------------------------------------------
    # MTP
    # ------------------------------------------------------------
    def has_mtp(self) -> bool:
        """False for now — transformers ships no MTP module for this arch.

        The checkpoint *does* carry a full sidecar under ``mtp.`` (see
        `mtp_source_prefix`), but `Qwen4ExpForCausalLM` drops it on load
        (``_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]``,
        modeling_qwen4_exp.py:1535) and no reference forward exists in HF or
        vLLM. Qwen3.5's `build_mtp_module()` could be written because its MTP
        block is an ordinary decoder layer assembled from HF primitives; a
        qwen4_exp replica would have to re-implement the hyper-connection
        decoder, the QSA indexer and the fused-expert MoE block from scratch —
        that is a reimplementation, not scaffolding, and it would be unattested.

        TODO(qwen4_exp-mtp): flip to True and implement `build_mtp_module()`
        once a reference MTP forward exists (HF module or vLLM class). The
        sidecar's inventory is recorded in the recon memo; the generic,
        index-driven `read_mtp_source_state_dict()` inherited from base already
        works against the ``mtp.`` prefix.
        """
        return False

    def mtp_source_prefix(self) -> str | None:
        """``"mtp."`` — the sidecar's source-key namespace.

        Index keys: ``mtp.fc_embedding.weight``, ``mtp.fc_hidden.weight``,
        ``mtp.pre_fc_norm_embedding.weight``, ``mtp.pre_fc_norm_hidden.weight``,
        ``mtp.hyper_connection_mixer.*``, ``mtp.layers.0.*``.

        Note the divergence from Qwen3.5/3.6, which store a single ``mtp.fc``:
        this family has **two** projection Linears (``fc_embedding`` and
        ``fc_hidden``), declared in the spec's `mtp_extra_linear_names`.

        Declared even though `has_mtp()` is False, so the walk's MTP exclusion
        rule and the source-passthrough prefix both resolve correctly and the
        sidecar's bytes are never silently folded into the quantizable body.
        """
        return "mtp."

    # ------------------------------------------------------------
    # Discovery-walk claims
    # ------------------------------------------------------------
    def walk_claim_rules(self):
        """Two matmul-fed / non-GEMM families the base rules cannot claim.

        `base.walk_claim_rules()` rule 9 only decides ``nn.Linear`` weights;
        an unclaimed *matmul-fed* node fails the walk by design
        (`model_walk.py:894-912` — the ``wo_a`` failure class). qwen4_exp has
        one such node and one adjacent case:

        1. **The MoE router.** `Qwen4ExpTextTopKRouter` holds a bare
           ``nn.Parameter`` of shape ``[num_experts, hidden]``
           (modeling_qwen4_exp.py:905) and calls ``F.linear(hidden_states,
           self.weight)`` in its forward — matmul-fed, 2-D, but **not** an
           ``nn.Linear``, so no base rule matches it. Pinned, following the
           DSv4 precedent (`deepseek_v4.py:450-460`): a route flip is not a
           smooth cost, so no surrogate in this codebase can price it.
        2. **The DeltaNet / PLE short depthwise convolutions.** ``conv1d.weight``
           is 3-D (e.g. ``[10240, 1, 4]`` for ``linear_attn.conv1d``) and is
           consumed by a convolution rather than a GEMM, so it is neither
           claimed by rule 9 nor excluded by rule 8's ``max_ndim=1``. Pinned
           explicitly rather than left unclaimed, so its bytes are named on the
           immutable floor instead of being silently uncategorised.

        Everything else falls to the base rules: the packed 3-D expert
        parameters are handled by the spec's `packed_experts` declaration, the
        QSA indexer projection and the shared-expert sigmoid gate are ordinary
        Linears pinned via `pinned_names()` (base rule 2), the PLE n-gram
        embedding shards are ``nn.Embedding`` (base rule 5), and the DeltaNet
        ``A_log``/``dt_bias`` parameters are 1-D (base rule 8).
        """
        from prismaquant.model_walk import ClaimRule

        rules = [
            ClaimRule(
                "pin",
                "MoE router gate: a bare nn.Parameter fed to F.linear "
                "(Qwen4ExpTextTopKRouter), matmul-fed but never priced — a "
                "route flip is not a smooth cost; held at source precision",
                module_class="Qwen4ExpTextTopKRouter",
            ),
            ClaimRule(
                "decide",
                "packed MoE expert stack: a 3-D nn.Parameter sliced per "
                "expert and fed to F.linear (Qwen4ExpTextExperts) — "
                "matmul-fed, not an nn.Linear, and 97% of this "
                "architecture's quantizable bytes; the allocator's domain",
                module_class="Qwen4ExpTextExperts",
            ),
            ClaimRule(
                "pin",
                "short depthwise convolution kernel (gated-DeltaNet / PLE): "
                "consumed by a convolution, not a GEMM; held at source "
                "precision on the immutable floor",
                name_regex=r"(?:^|\.)conv1d\.weight$",
            ),
        ]
        return rules + super().walk_claim_rules()
