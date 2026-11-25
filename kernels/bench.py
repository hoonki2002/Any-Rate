"""Batch-1 decode GEMV time of every linear-layer shape of Llama-3.1-8B: FP16 (cuBLAS), AP, AB, MG, MG+perm and Any-Rate.

  python -m kernels.bench          -> kernel_results/validation.json, gemv_llama.json   (needs `bash kernels/build.sh` and `bash baselines/setup.sh kernels`)

Every baseline is the released code path of its authors: AP the Any-Precision kernel (the copy in the AnyBCQ repository, which adds 2 bit;
the official build, 3-8 bit only, is timed against it with the profiler), AB AnyBCQLinear, MG MatGPTQLinear (8-bit master, per-group
zeros), MG+perm with the act_order input gather of its released inference path. Any-Rate: the fastest kernel and launch configuration
(kernels/ar_gemv.py) per shape and mode, tuned with short runs. Timing: one CUDA graph calls n independent weight copies in turn
(>= 320 MB, so the weights come from DRAM); R = 5 rounds, every (method, bits) once per round in random order; reported: the median
over rounds of graph time / n. Inputs and weights are random (the time does not depend on the values). Before timing, each baseline
kernel is checked against its own dense path (validation)."""
import argparse, copy, json, os, random, statistics, sys
if __name__ == "__main__": argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()   # --help before the imports below
import torch
import torch.nn.functional as F

T = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "baselines", "third_party")
sys.path[:0] = [os.path.join(T, "anybcq"), os.path.join(T, "anybcq", "anybcq", "inference", "custom_kernel"), os.path.join(T, "MatGPTQ", "inference_lib"),
                os.path.join(T, "any-precision-llm", "any_precision", "modules", "kernels")]       # built in place by baselines/setup.sh kernels
from anybcq.inference.AnyBCQLinear import AnyBCQLinear
import custom_kernel
import any_precision_ext as ap_official
from matgptq_cuda.matgptq_linear import MatGPTQLinear
from .ar_gemv import ARGemv

DEV, G, R, TARGET = "cuda", 128, 5, 320e6
SHAPES = {"q/o": (4096, 4096), "k/v": (1024, 4096), "gate/up": (14336, 4096), "down": (4096, 14336)}
AR_BITS = {3: 2, 4: 2.5, 5: 3, 6: 3.5, 7: 4}


def timed(calls, reps=25):
    """us per call: one CUDA graph over all calls, median of `reps` replays"""
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for f in calls: f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in calls: f()
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    del g; return statistics.median(ts) / len(calls) * 1e3


def ncopies(nbytes): return max(4, int(TARGET // nbytes) + 1)
def rel(y, r): return ((y.float() - r.float()).abs().max() / r.float().abs().max()).item()


def synth_ar(M, K, gen):
    """random tensors of one Any-Rate layer (checkpoint format)"""
    return {"plane0": torch.randint(0, 256, (M, K // 8), dtype=torch.uint8, device=DEV, generator=gen),
            **{f"plane{p}": torch.randint(0, 256, (M, K // 16), dtype=torch.uint8, device=DEV, generator=gen) for p in range(1, 7)},
            "table_codes": torch.randint(0, 256, (M, 128), dtype=torch.uint8, device=DEV, generator=gen).sort(dim=1).values,
            "table_lo": (-0.05 * torch.ones(M, 1, device=DEV)).half(), "table_step": (0.1 / 255 * torch.ones(M, 1, device=DEV)).half()}


def ar_candidates(K):
    c = []
    for W in (8, 16, 32):
        for S in (1, 2, 4, 8):
            if (K // 64) // S < W: continue
            if W < 32: c.append(f"v8 S={S} W={W}")
            c += [f"v11 S={S} W={W}", f"v11d S={S} W={W}"]
    return c + [f"{v} S=0 W={W}" for W in (8, 16, 32) for v in ("v12", "v14", "v14d")]


def parse(c):
    name, *kv = c.split(); kw = {a: int(b) for a, b in (t.split("=") for t in kv)}; kw[name.rstrip("d")] = True
    if name.endswith("d"): kw["dense"] = True
    return kw


def tune_ar(ar, x, k):
    """the fastest kernel and launch configuration for one shape and mode: short runs, then the three best re-timed"""
    run = lambda c: timed([(lambda m=m: m(x, k, **parse(c))) for m in ar], reps=6)
    best3 = sorted(ar_candidates(x.shape[1]), key=run)[:3]
    return min(best3, key=lambda c: statistics.median(timed([(lambda m=m: m(x, k, **parse(c))) for m in ar], reps=10) for _ in range(3)))


def validate(M=4096, K=4096):
    torch.manual_seed(0); out = {}; x = torch.randn(1, K, device=DEV).half()
    qw = torch.randint(-2**31, 2**31 - 1, (8, M, K // 32), dtype=torch.int32, device=DEV)
    for b in (3, 4):
        lut = (torch.randn(M, 2 ** b, device=DEV) * 0.02).half(); y = ap_official.matmul_kbit(x, qw, lut, b)
        out[f"AP official {b}b gemv vs own dequant"] = rel(y, x.float() @ ap_official.dequant_kbit(qw, lut, b).float().T)
        o = torch.zeros(1, 1, M, device=DEV).half(); custom_kernel.anyprec_gemv(x.view(1, 1, K), o, qw[:4].contiguous(), lut, b)
        out[f"AP port {b}b vs AP official"] = rel(o.view(1, M), y)
    m = AnyBCQLinear(K, M, [2, 3, 4], G, dtype=torch.half).cuda(); m.qweight.random_(-2**31, 2**31 - 1); m.q_bias.normal_(0, 0.01)
    for b in (2, 3, 4): m._get_alpha(b).uniform_(0.005, 0.02); m._get_beta(b).normal_(0, 0.01)
    for b in (2, 3, 4):
        m.set_precision(b); y = m(x.view(1, 1, K)).clone()
        out[f"AB {b}b gemv vs own dequant+matmul"] = rel(y.view(1, M), m._gemm(x.view(1, 1, K), b, m._get_alpha(b), m._get_beta(b)).view(1, M))
    codes = torch.randint(0, 256, (M, K), dtype=torch.uint8); scales = (torch.rand(M, K // G) * 0.01 + 0.002).half(); zeros = torch.randint(100, 156, (M, K // G)).half()
    mg = MatGPTQLinear(codes, scales, 8, G, zeros=zeros).cuda()
    for b in (2, 3, 4):
        mg.bitwidth = b; cb = (codes.int() >> (8 - b)).float() * 2 ** (8 - b) + (2 ** (8 - b) - 1) / 2
        out[f"MG {b}b (zeros) vs dense formula"] = rel(mg(x), x.float() @ ((cb.view(M, K // G, G) - zeros.float()[..., None]) * scales.float()[..., None]).view(M, K).to(DEV).T)
    return out


def cupti_us(fns, name_filter, reps=3):
    """mean kernel time (us) from the profiler, for kernels whose name contains name_filter (the official AP launch cannot be graph-captured)"""
    for f in fns: f()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            for f in fns: f()
        torch.cuda.synchronize()
    ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA and name_filter in e.name]
    return sum(e.device_time for e in ev) / max(1, len(ev))


def run_shape(M, K):
    torch.manual_seed(0); gen = torch.Generator(device=DEV).manual_seed(0); x = torch.randn(1, K, device=DEV).half(); x3 = x.view(1, 1, K); calls = {}
    Ws = [torch.randn(M, K, device=DEV).half() * 0.02 for _ in range(ncopies(M * K * 2))]; calls["fp16"] = [(lambda W=W: F.linear(x, W)) for W in Ws]
    qws = [torch.randint(-2**31, 2**31 - 1, (4, M, K // 32), dtype=torch.int32, device=DEV, generator=gen) for _ in range(ncopies(M * K / 2))]; o = torch.zeros(1, 1, M, device=DEV).half()
    for b in (2, 3, 4):
        luts = [(torch.randn(M, 2 ** b, device=DEV) * 0.02).half() for _ in qws]
        calls[f"AP {b}"] = [(lambda q=q, l=l, b=b: custom_kernel.anyprec_gemv(x3, o, q, l, b)) for q, l in zip(qws, luts)]
    abm = []
    for _ in range(ncopies(M * K * 4.5 / 8)):
        m = AnyBCQLinear(K, M, [2, 3, 4], G, dtype=torch.half).cuda(); m.qweight.random_(-2**31, 2**31 - 1, generator=gen); m.q_bias.normal_(0, 0.01)
        for b in (2, 3, 4): m._get_alpha(b).uniform_(0.005, 0.02); m._get_beta(b).normal_(0, 0.01)
        abm.append(m)
    for b in (2, 3, 4):
        ms = [copy.copy(m) for m in abm]
        for m in ms: m.set_precision(b)
        calls[f"AB {b}"] = [(lambda m=m: m(x3)) for m in ms]
    codes = torch.randint(0, 256, (M, K), dtype=torch.uint8); scales = (torch.rand(M, K // G) * 0.01 + 0.002).half(); zeros = torch.randint(100, 156, (M, K // G)).half()
    proto = MatGPTQLinear(codes, scales, 8, G, zeros=zeros).cuda(); del codes
    n = ncopies(M * K / 2); mgm = [proto] + [copy.deepcopy(proto) for _ in range(n - 1)]; perms = [torch.randperm(K, device=DEV) for _ in range(n)]
    for b in (2, 3, 4):
        ms = [copy.copy(m) for m in mgm]
        for m in ms: m.bitwidth = b
        calls[f"MG {b}"] = [(lambda m=m: m(x)) for m in ms]; calls[f"MG+perm {b}"] = [(lambda m=m, p=p: m(x[:, p])) for m, p in zip(ms, perms)]
    ar = [ARGemv(synth_ar(M, K, gen)) for _ in range(ncopies(M * K / 2 + M * 132))]; cfg = {}
    for k, bits in AR_BITS.items():
        cfg[bits] = tune_ar(ar, x, k); calls[f"AR {bits}"] = [(lambda m=m, kw=parse(cfg[bits]), k=k: m(x, k, **kw)) for m in ar]
    keys = list(calls); res = {k: [] for k in keys}
    for r in range(R):
        random.Random(r).shuffle(keys)
        for k in keys: res[k].append(timed(calls[k], reps=10))
    med = {k: round(statistics.median(v), 2) for k, v in res.items()}; spr = {k: round(max(v) - min(v), 2) for k, v in res.items()}
    row = {"M": M, "K": K, "fp16_us": med.pop("fp16"), "fp16_spread_us": spr.pop("fp16"), "us": med, "spread_us": spr,
           "ar_kernel": {f"{b:g}": c for b, c in cfg.items()}, "cupti_ap_us": {}}
    for b in (3, 4):
        luts = [(torch.randn(M, 2 ** b, device=DEV) * 0.02).half() for _ in qws]
        row["cupti_ap_us"][f"official {b}"] = round(cupti_us([(lambda q=q, l=l: ap_official.matmul_kbit(x, q, l, b)) for q, l in zip(qws, luts)], "matmul_kbit"), 2)
        row["cupti_ap_us"][f"copy {b}"] = round(cupti_us([(lambda q=q, l=l: custom_kernel.anyprec_gemv(x3, o, q, l, b)) for q, l in zip(qws, luts)], "matmul_kbit"), 2)
    return row


def main():
    """kernel_results/validation.json and gemv_llama.json, in the format of results/H.Storage_and_Decoding_Efficiency/kernel_latency"""
    os.makedirs("kernel_results", exist_ok=True)
    json.dump({"max relative error of each baseline kernel against its own dense path": validate()}, open(os.path.join("kernel_results", "validation.json"), "w"), indent=1)
    out = {"gpu": torch.cuda.get_device_name(0), "model": "Llama-3.1-8B-Instruct", "batch": 1, "statistic": f"median of {R} rounds", "rows": []}
    for name, (M, K) in SHAPES.items():
        out["rows"].append({"layer": name, **run_shape(M, K)}); json.dump(out, open(os.path.join("kernel_results", "gemv_llama.json"), "w"), indent=1)
        print(name, out["rows"][-1]["us"], flush=True); torch.cuda.empty_cache()
    N = {"q/o": 2, "k/v": 2, "gate/up": 2, "down": 1}               # per decoder layer (q, k, v, o, gate, up, down); 32 layers
    print("linear layers per token (ms):", {m: round(32 * sum(N[r["layer"]] * r["us"][m] for r in out["rows"]) / 1e3, 2) for m in out["rows"][0]["us"]})

if __name__ == "__main__":
    main()
