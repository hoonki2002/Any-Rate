"""Batch-1 decoding throughput (tokens/s) of Llama-3.1-8B with each method's linear layers.

  python -m kernels.tps            -> kernel_results/tps_llama.json   (after python -m kernels.bench, whose Any-Rate configurations it uses)

One decoder written once (32 layers, GQA 32/8 heads x 128, SwiGLU 14336, RMSNorm, RoPE, KV cache; fp16 embedding, attention and lm_head);
only the projections differ, each the released linear path as in bench.py (q/k/v and gate/up fused, as in the AnyBCQ gpt-fast model and
vLLM); "none": projections return a fixed buffer (the cost of everything else). MG is timed without and with (MG+perm) the act_order
gather. Weights are random; one decode step is captured in a CUDA graph (argmax feeds the next token); tokens/s = 256 / time of 256
replays from position 128. Two passes over the methods in opposite order, 3 rounds each with the bit widths in random order; median."""
import argparse, copy, json, os, random, statistics
if __name__ == "__main__": argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()   # --help before the imports below
import torch
import torch.nn.functional as F
from .bench import AnyBCQLinear, MatGPTQLinear, custom_kernel, ARGemv, synth_ar, ncopies, tune_ar, parse

DEV, G = "cuda", 128; torch.set_grad_enabled(False)
D, NL, NH, NKV, HD, FF, V = 4096, 32, 32, 8, 128, 14336, 128256
P0, NTOK, LMAX, R = 128, 256, 512, 3
AR_K = {2: 3, 2.5: 4, 3: 5, 3.5: 6, 4: 7}
BITS = {"none": [0], "AP": [2, 3, 4], "AB": [2, 3, 4], "MG": [2, 3, 4], "AR": [2, 2.5, 3, 3.5, 4]}
PROJ = {"qkv": (NH * HD + 2 * NKV * HD, D), "o": (D, NH * HD), "gate_up": (2 * FF, D), "down": (D, FF)}


def ar_configs():
    rows = {r["layer"]: r["ar_kernel"] for r in json.load(open(os.path.join("kernel_results", "gemv_llama.json")))["rows"]}; cfg = {}
    for name, (M, K) in PROJ.items():
        if name in ("o", "down"): cfg[name] = {float(b): c for b, c in rows["q/o" if name == "o" else "down"].items()}; continue
        gen = torch.Generator(device=DEV).manual_seed(1); x = torch.randn(1, K, device=DEV).half()
        ar = [ARGemv(synth_ar(M, K, gen)) for _ in range(ncopies(M * K / 2 + M * 132))]
        cfg[name] = {float(b): tune_ar(ar, x, k) for b, k in AR_K.items()}; del ar; torch.cuda.empty_cache(); print("tuned", name, cfg[name], flush=True)
    return cfg


class Base:
    def __init__(self):
        g = torch.Generator(device=DEV).manual_seed(0)
        self.embed = (torch.randn(V, D, device=DEV, generator=g) * 0.02).half(); self.lm_head = (torch.randn(V, D, device=DEV, generator=g) * 0.02).half()
        self.ln1 = torch.ones(NL, D, device=DEV).half(); self.ln2 = torch.ones(NL, D, device=DEV).half(); self.lnf = torch.ones(D, device=DEV).half()
        self.kc = torch.zeros(NL, 1, NKV, LMAX, HD, device=DEV).half(); self.vc = torch.zeros_like(self.kc)
        f = torch.outer(torch.arange(LMAX, device=DEV).float(), 1.0 / 500000 ** (torch.arange(0, HD, 2, device=DEV).float() / HD)); self.cos, self.sin = f.cos().half(), f.sin().half()
        self.mask = torch.ones(LMAX, LMAX, dtype=torch.bool, device=DEV).tril()
        self.tok = torch.zeros(1, dtype=torch.long, device=DEV); self.pos = torch.zeros(1, dtype=torch.long, device=DEV); self.logits = torch.zeros(1, V, device=DEV).half()


def rope(x, c, s):
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]; return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)


def step(B, L, b):
    h = B.embed.index_select(0, B.tok); c = B.cos.index_select(0, B.pos).view(1, 1, 1, -1); s = B.sin.index_select(0, B.pos).view(1, 1, 1, -1)
    m = B.mask.index_select(0, B.pos).view(1, 1, 1, LMAX)
    for l in range(NL):
        q, k, v = L[l]["qkv"](F.rms_norm(h, (D,), B.ln1[l], 1e-5), b).split([NH * HD, NKV * HD, NKV * HD], -1)
        q = rope(q.reshape(1, 1, NH, HD).transpose(1, 2), c, s); k = rope(k.reshape(1, 1, NKV, HD).transpose(1, 2), c, s)
        B.kc[l].index_copy_(2, B.pos, k); B.vc[l].index_copy_(2, B.pos, v.reshape(1, 1, NKV, HD).transpose(1, 2))
        o = F.scaled_dot_product_attention(q, B.kc[l], B.vc[l], attn_mask=m, enable_gqa=True)
        h = h + L[l]["o"](o.transpose(1, 2).reshape(1, D).contiguous(), b)
        g, u = L[l]["gate_up"](F.rms_norm(h, (D,), B.ln2[l], 1e-5), b).split([FF, FF], -1)
        h = h + L[l]["down"]((F.silu(g) * u).contiguous(), b)
    B.logits.copy_(F.linear(F.rms_norm(h, (D,), B.lnf, 1e-5), B.lm_head)); B.tok.copy_(B.logits.argmax(-1)); B.pos.add_(1)


def build(meth, arcfg):
    """projections f(x [1, K], bits) -> [1, M] of one method, random weights"""
    gen = torch.Generator(device=DEV).manual_seed(2); L = [dict() for _ in range(NL)]
    for name, (M, K) in PROJ.items():
        if meth == "none":
            y = torch.zeros(1, M, device=DEV).half()
            for l in range(NL): L[l][name] = (lambda x, b, y=y: y)
        elif meth == "AP":
            for l in range(NL):
                q = torch.randint(-2**31, 2**31 - 1, (4, M, K // 32), dtype=torch.int32, device=DEV, generator=gen)
                lut = {b: (torch.randn(M, 2 ** b, device=DEV, generator=gen) * 0.02).half() for b in (2, 3, 4)}; o = torch.empty(1, 1, M, device=DEV).half()
                def f(x, b, q=q, lut=lut, o=o, K=K, M=M): custom_kernel.anyprec_gemv(x.view(1, 1, K), o, q, lut[b], b); return o.view(1, M)
                L[l][name] = f
        elif meth == "AB":
            proto = AnyBCQLinear(K, M, [2, 3, 4], G, dtype=torch.half).cuda()
            for l in range(NL):
                m = proto if l == 0 else copy.deepcopy(proto); m.output = torch.zeros(1, 1, M, device=DEV).half()
                m.qweight.random_(-2**31, 2**31 - 1, generator=gen); m.q_bias.normal_(0, 0.01, generator=gen)
                for b in (2, 3, 4): m._get_alpha(b).uniform_(0.005, 0.02, generator=gen); m._get_beta(b).normal_(0, 0.01, generator=gen)
                L[l][name] = (lambda x, b, m=m, K=K, M=M: m(x.view(1, 1, K), precision=b).view(1, M))
        elif meth == "MG":       # b < 0: MG+perm (x gathered by the layer's act_order permutation first)
            codes = torch.randint(0, 256, (M, K), dtype=torch.uint8); scales = (torch.rand(M, K // G) * 1e-4 + 2e-5).half(); zeros = torch.randint(100, 156, (M, K // G)).half()
            proto = MatGPTQLinear(codes, scales, 8, G, zeros=zeros).cuda(); del codes
            for l in range(NL):
                m = proto if l == 0 else copy.deepcopy(proto); p = torch.randperm(K, device=DEV, generator=gen)
                def f(x, b, m=m, p=p): m.bitwidth = abs(b); return m(x[:, p] if b < 0 else x)
                L[l][name] = f
        elif meth == "AR":
            kws = {b: parse(c) for b, c in arcfg[name].items()}
            for l in range(NL):
                m = ARGemv(synth_ar(M, K, gen)); L[l][name] = (lambda x, b, m=m, kws=kws: m(x, AR_K[b], **kws[float(b)]))
    return L


def capture(B, L, b):
    cur = torch.cuda.current_stream(); s = torch.cuda.Stream(); s.wait_stream(cur); B.pos.fill_(P0); B.tok.zero_()
    with torch.cuda.stream(s):
        for _ in range(3): step(B, L, b)
    cur.wait_stream(s); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): step(B, L, b)
    return g


def run(B, g):
    B.pos.fill_(P0); torch.cuda.synchronize(); a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); a.record()
    for _ in range(NTOK): g.replay()
    e.record(); torch.cuda.synchronize(); return NTOK / (a.elapsed_time(e) / 1e3)


def main():
    """kernel_results/tps_llama.json, in the format of results/H.Storage_and_Decoding_Efficiency/throughput"""
    arcfg = ar_configs(); B = Base(); tps = {}; f = os.path.join("kernel_results", "tps_llama.json")
    out = {"gpu": torch.cuda.get_device_name(0), "model": "Llama-3.1-8B-Instruct (random weights in each format)", "batch": 1, "fused_projections": PROJ,
           "start_position": P0, "tokens": NTOK, "statistic": f"median of {2 * R} runs (two passes x {R} rounds)", "note": "'none 0' = the decoder without its linear layers",
           "tokens_per_s": {}, "ar_kernel": {n: {f"{b:g}": c for b, c in v.items()} for n, v in arcfg.items()}}
    for order in (["none", "AP", "AB", "MG", "AR"], ["AR", "MG", "AB", "AP", "none"]):
        for meth in order:
            L = build(meth, arcfg)
            variants = [("MG", b) for b in BITS["MG"]] + [("MG+perm", -b) for b in BITS["MG"]] if meth == "MG" else [(meth, b) for b in BITS[meth]]
            graphs = {(n, b): capture(B, L, b) for n, b in variants}
            for _ in range(R):
                keys = list(graphs); random.shuffle(keys)
                for key in keys: run(B, graphs[key]); tps.setdefault(f"{key[0]} {abs(key[1]):g}", []).append(run(B, graphs[key]))
            del graphs, L; torch.cuda.empty_cache()
            out["tokens_per_s"] = {k: {"median": round(statistics.median(v), 2), "all": [round(t, 2) for t in v]} for k, v in sorted(tps.items())}
            json.dump(out, open(f, "w"), indent=1)
    print(json.dumps({k: v["median"] for k, v in out["tokens_per_s"].items()}))

if __name__ == "__main__":
    main()
