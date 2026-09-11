#!/usr/bin/env python3
"""Expert routing histogram of the FULL (unquantized) model over a text corpus.

Streams the model layer by layer (the probe's phase-1 machinery) and counts,
per router, how often each expert is selected and how much softmax mass it
receives -- the expert map a frequency-tiered artifact is built from, measured
on the source weights rather than on the quantized engine.

Layer-major over a chunk of batches: install layer L once, push every batch
of the chunk through it, move on.  On a discrete GPU the chunk's hidden states
(one per batch, ~1 GB each at 32 x 1024 x 4 x 4096 bf16) stay on the device.

Hash-routed layers (DeepseekV4HashRouter) select by the frozen `tid2eid`
lookup, not by their scores; their histogram is computed from the token ids
directly and the hook counts are not reported for them.

usage: route_histogram.py --model DIR --texts corpus.jsonl --out hist.json
       [--seqlen 1024] [--batch 32] [--chunk 12] [--shard i --nshards n]
       [--device cuda] [--cache-headroom-gb 60]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch


def build_windows(texts_path: str, tokenizer, seqlen: int) -> tuple[torch.Tensor, int]:
    """Consecutive non-overlapping `seqlen` windows over the corpus in file
    order, documents joined by EOS; the tail shorter than `seqlen` is dropped.
    Returns (windows [N, seqlen] long, total tokens seen)."""
    eos = tokenizer.eos_token_id
    buf: list[int] = []
    windows: list[torch.Tensor] = []
    seen = 0
    with open(texts_path) as f:
        for line in f:
            row = json.loads(line)
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            ids = tokenizer(text, add_special_tokens=False, truncation=False).input_ids
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            seen += len(ids)
            buf.extend(int(v) for v in ids)
            if eos is not None:
                buf.append(int(eos))
            while len(buf) >= seqlen:
                windows.append(torch.tensor(buf[:seqlen], dtype=torch.long))
                del buf[:seqlen]
    if not windows:
        sys.exit("no windows built")
    return torch.stack(windows, dim=0), seen


def _load_tid2eid_from_checkpoint(model_path: str, router_qname: str) -> torch.Tensor:
    """`layers.N.ffn.gate.tid2eid` ([vocab, top_k] int64) straight from the safetensors shard that holds it."""
    import struct
    layer = router_qname.split(".")[2]
    name = f"layers.{layer}.ffn.gate.tid2eid"
    with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
        shard = json.load(f)["weight_map"][name]
    with open(os.path.join(model_path, shard), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        meta = header[name]
        if meta["dtype"] != "I64":
            raise RuntimeError(f"{name}: expected I64, got {meta['dtype']}")
        start, end = meta["data_offsets"]
        f.seek(8 + n + start)
        raw = f.read(end - start)
    return torch.frombuffer(bytearray(raw), dtype=torch.int64).reshape(meta["shape"]).clone()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--texts", required=True, help="jsonl of {\"text\": ...} rows (chat cases already rendered)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=12, help="batches per layer pass (hidden states resident on the device)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--cache-headroom-gb", type=float, default=60.0)
    ap.add_argument("--work-dir", default="/tmp/route-histogram-work")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from prismaquant.sensitivity_probe import (
        RouterTracker, discover_moe_routers, read_top_k, stage_text_only)
    from prismaquant.streaming_model import _build_streaming_context
    from prismaquant.layer_streaming import (
        _call_layer, _compute_attention_mask, _compute_position_embeddings)
    from prismaquant.model_profiles import profile_from_model

    device = torch.device(a.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]

    staged = stage_text_only(a.model)
    tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)
    t0 = time.time()
    windows, seen = build_windows(a.texts, tokenizer, a.seqlen)
    windows = windows[a.shard::a.nshards]
    print(f"[route] {seen:,} corpus tokens -> {windows.size(0)} windows x {a.seqlen} for shard {a.shard}/{a.nshards} "
          f"({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(a.work_dir, exist_ok=True)
    ctx = _build_streaming_context(
        a.model, device=device, dtype=dtype, offload_folder=os.path.join(a.work_dir, "offload"),
        cache_headroom_gb=a.cache_headroom_gb, log_prefix="[route]")
    model, base_model, layers, num_layers = ctx.model, ctx.base_model, ctx.layers, ctx.num_layers
    profile = profile_from_model(model)
    routers = sorted(discover_moe_routers(model, profile=profile))
    top_k = read_top_k(model)
    tracker = RouterTracker(model, routers, top_k=top_k)
    # Hash routers select by a frozen `tid2eid` table. The module's buffer is
    # only materialized when its layer is installed (it is zeros on the meta
    # skeleton -- the 2026-09-11 run counted every hash-layer token as expert
    # 0), so the table is read from the checkpoint itself.
    hash_routers: dict[str, torch.Tensor] = {}
    for rq in routers:
        mod = model.get_submodule(rq)
        if type(mod).__name__ == "DeepseekV4HashRouter":
            hash_routers[rq] = _load_tid2eid_from_checkpoint(a.model, rq)
    n_experts = {rq: int(tracker.counts_t[rq].numel()) for rq in routers if rq in tracker.counts_t}
    hash_counts = {rq: torch.zeros(n_experts[rq], dtype=torch.int64) for rq in hash_routers}
    print(f"[route] {len(routers)} routers ({len(hash_routers)} hash-routed), top_k={top_k}, "
          f"{num_layers} layers, batch {a.batch}, chunk {a.chunk}", flush=True)

    n_windows = windows.size(0)
    batches = [windows[i:i + a.batch] for i in range(0, n_windows, a.batch)]
    tokens_done = 0
    t_start = time.time()
    for c0 in range(0, len(batches), a.chunk):
        chunk = batches[c0:c0 + a.chunk]
        t_chunk = time.time()
        with torch.no_grad():
            ids_list = [b.to(device) for b in chunk]
            hiddens = []
            for ids in ids_list:
                h = base_model.embed_tokens(ids).to(dtype)
                hiddens.append(h)
            # position embeddings / mask depend on the window shape only
            position_ids = torch.arange(a.seqlen, device=device).unsqueeze(0)
            pos_emb = _compute_position_embeddings(base_model, hiddens[0], position_ids, profile)
            mask = _compute_attention_mask(base_model, hiddens[0], position_ids)
            hiddens = [profile.expand_hidden_for_layers(h, base_model) for h in hiddens]
            states = [profile.new_forward_pass_state() for _ in chunk]
            # hash-routed layers: the true selection is tid2eid[token]
            for rq, table in hash_routers.items():
                tbl = table.to("cpu") if table.device.type != "cpu" else table
                for ids in ids_list:
                    sel = tbl[ids.to("cpu").reshape(-1)].reshape(-1)
                    hash_counts[rq] += torch.bincount(sel, minlength=n_experts[rq])
            ctx.schedule_prefetch(0)
            for L in range(num_layers):
                t_l = time.time()
                src = ctx.install(L)
                ctx.schedule_prefetch(L + 1)
                load_s = time.time() - t_l
                t_f = time.time()
                for bi, ids in enumerate(ids_list):
                    hiddens[bi] = _call_layer(
                        layers[L], hiddens[bi], position_embeddings=pos_emb, attention_mask=mask,
                        position_ids=position_ids, **profile.extra_layer_kwargs(input_ids=ids),
                        pass_state=states[bi])
                fwd_s = time.time() - t_f
                ctx.unload(L)
                if L % 8 == 0 or L == num_layers - 1:
                    print(f"[route]   L{L:02d} src={src} load={load_s:.2f}s fwd={fwd_s:.2f}s ({len(chunk)} batches)", flush=True)
            del hiddens, states
        tokens_done += sum(b.numel() for b in chunk)
        el = time.time() - t_start
        print(f"[route] chunk {c0 // a.chunk + 1}/{(len(batches) + a.chunk - 1) // a.chunk}: {tokens_done:,} tokens, "
              f"{time.time()-t_chunk:.0f}s, {tokens_done/el:,.0f} tok/s overall", flush=True)
        _write(a, routers, hash_routers, hash_counts, tracker, n_experts, tokens_done, n_windows, top_k, num_layers, partial=True)
    tracker.remove_hooks()
    _write(a, routers, hash_routers, hash_counts, tracker, n_experts, tokens_done, n_windows, top_k, num_layers, partial=False)
    print(f"[route] wrote {a.out} ({tokens_done:,} tokens in {time.time()-t_start:.0f}s)", flush=True)


def _write(a, routers, hash_routers, hash_counts, tracker, n_experts, tokens_done, n_windows, top_k, num_layers, *, partial):
    out = {"model": a.model, "texts": a.texts, "seqlen": a.seqlen, "shard": a.shard, "nshards": a.nshards,
           "windows": int(n_windows), "tokens": int(tokens_done), "top_k": int(top_k), "layers": int(num_layers),
           "partial": partial, "routers": {}}
    for rq in routers:
        if rq not in n_experts:
            continue
        entry = {"n_experts": n_experts[rq]}
        if rq in hash_routers:
            entry["kind"] = "hash"
            entry["active_counts"] = hash_counts[rq].tolist()
            entry["total_tokens"] = int(hash_counts[rq].sum().item() // max(1, top_k))
        else:
            entry["kind"] = "learned"
            entry["active_counts"] = tracker.active_counts_t[rq].tolist()
            entry["prob_mass"] = [round(v, 6) for v in tracker.counts_t[rq].tolist()]
            entry["total_tokens"] = int(tracker.total_tokens.get(rq, 0))
        out["routers"][rq] = entry
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
