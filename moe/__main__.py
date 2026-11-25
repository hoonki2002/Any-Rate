"""Qwen3-30B-A3B-Instruct-2507 with Any-Rate experts (Appendix I): per-expert modes and on-device decoding speed.

  python -m moe allocation     per-expert modes at mean bit widths 2.5, 3 and 3.5 (moe_results/allocation, as results/I.../allocation)
  python -m moe speed          decoding speed with a VRAM expert cache (moe_results/speed, as results/I.../speed)

allocation: one calibration pass (the C4 set of the encoding) records the gate mass of every expert (router softmax, top-8, renormalised)
and the relative loss tr(E H E^T) / tr(W H W^T) of every expert matrix in every mode of the Any-Rate checkpoint (raw Hessian; identity for
an expert that receives no token). Per layer, with the mean bit width b of its 128 experts fixed, the modes minimise
sum_e g_e err_e(mode_e), g = normalised gate mass, err_e = mean relative loss of its three matrices (greedy half-bit steps: gate_mass);
reverse gives the same modes in reverse gate-mass order. python -m evaluation --model qwen3-30b-a3b-instruct-2507 --method ar_gate_mass
evaluates them.
speed: the non-expert weights (attention, router, norms, embeddings, lm_head) stay in VRAM in bf16; every layer keeps an LRU cache of
experts in VRAM and reads a missing one from the NVMe with O_DIRECT (engine.py). BF16 experts are read from the original shards, Any-Rate
experts of mode K from files repacked so that a mode is one read (repack). 3 prompts x 64 generated tokens per point; caches 2-10 GB."""
import argparse, glob, heapq, json, os, subprocess, sys, time
import numpy as np
import torch
from safetensors import safe_open

E, NL, MODES = 128, 48, np.array([2.0, 2.5, 3.0, 3.5, 4.0])
CACHES = [2, 3, 4, 6, 8, 10]
log = lambda *x: print(time.strftime("[%H:%M:%S]"), *x, flush=True)


def ar_tensors(ckpt):
    """{layer name: {field: (file, key)}} of an Any-Rate checkpoint"""
    out = {}
    for f in sorted(glob.glob(os.path.join(ckpt, "*.safetensors"))):
        for k in safe_open(f, "pt").keys():
            if ".ar." in k: n, field = k.split(".ar."); out.setdefault(n, {})[field] = (f, k)
    return out


def load(ref, dev="cpu", fields=None):
    t = {}
    for field, (f, k) in ref.items():
        if fields is None or field in fields:
            with safe_open(f, "pt", device=dev) as fh: t[field] = fh.get_tensor(k)
    return t


def stats(src, ckpt, out):
    from any_rate.format import decode_planes
    from any_rate.models import decoder
    os.makedirs(out, exist_ok=True); todo = [li for li in range(NL) if not os.path.exists(os.path.join(out, f"l{li}_proxy.json"))]
    if not todo: return
    refs = ar_tensors(ckpt); gates = {}

    def on_layer(li, layer):
        cfg = layer.mlp.gate.out_features; k = 8; G = gates[li] = {"count": torch.zeros(cfg, device="cuda"), "mass": torch.zeros(cfg, device="cuda"), "tokens": 0}
        def rhook(mod, inp, o):
            w = torch.softmax(o.float().reshape(-1, o.shape[-1]), -1); tw, ti = torch.topk(w, k, -1); tw = tw / tw.sum(-1, keepdim=True)   # norm_topk_prob
            G["count"].index_add_(0, ti.reshape(-1), torch.ones(ti.numel(), device="cuda")); G["mass"].index_add_(0, ti.reshape(-1), tw.reshape(-1)); G["tokens"] += w.shape[0]
        return [layer.mlp.gate.register_forward_hook(rhook)]

    def save(li, store):
        prox = {}
        for name, t in store.items():
            W, H = t["W"], t["H"]; den = float(((W @ H) * W).sum()); P = load(refs[name], "cuda", {f"plane{p}" for p in range(7)} | {"table_codes", "table_lo", "table_step"})
            short = name.split(f"layers.{li}.")[1]; prox[short] = {}
            for k in range(3, 8):
                Ek = W - decode_planes(P, "UHHHHHH", k); prox[short][str(k)] = float(((Ek @ H) * Ek).sum()) / den
            prox[short + "#den"] = den
        G = gates.pop(li); json.dump({"count": G["count"].tolist(), "mass": G["mass"].tolist(), "tokens": G["tokens"], "top_k": 8}, open(os.path.join(out, f"l{li}_gate.json"), "w"))
        json.dump(prox, open(os.path.join(out, f"l{li}_proxy.json"), "w")); log(f"layer {li}: gate statistics and loss table")

    decoder.encode(src, lambda W, H: {"W": W, "H": H}, save, layers=todo, moe_prior=0, on_layer=on_layer, log=log)


def allocate(sd, out):
    def greedy(w, eps, steps):
        idx = np.zeros(E, int); h = [(-w[e] * (eps[e, 0] - eps[e, 1]), e) for e in range(E)]; heapq.heapify(h)
        for _ in range(steps):
            _, e = heapq.heappop(h); idx[e] += 1
            if idx[e] < 4: heapq.heappush(h, (-w[e] * (eps[e, idx[e]] - eps[e, idx[e] + 1]), e))
        return idx
    tokens = {}
    for budget, steps in ((3.5, 3 * E), (3.0, 2 * E), (2.5, E)):
        allocs = {"greedy": {}, "reverse": {}}
        for li in range(NL):
            g = json.load(open(os.path.join(sd, f"l{li}_gate.json"))); p = json.load(open(os.path.join(sd, f"l{li}_proxy.json")))
            gs = np.array(g["mass"]) / sum(g["mass"]); rel = np.zeros((E, 5))
            for e in range(E):
                for pr in ("gate_proj", "up_proj", "down_proj"): rel[e] += np.array([p[f"mlp.experts.{e}.{pr}"][str(k)] for k in range(3, 8)]) / 3
            gr = greedy(gs, rel, steps); rv = np.zeros_like(gr); rv[np.argsort(gs)] = np.sort(gr)[::-1]     # reverse: the lowest gate mass gets the most bits
            for m, ix in (("greedy", gr), ("reverse", rv)):
                assert abs(MODES[ix].mean() - budget) < 1e-9; allocs[m][str(li)] = {str(e): float(MODES[ix[e]]) for e in range(E)}
            tokens[str(li)] = {f"e{e}{s}": int(g["count"][e]) for e in range(E) for s in ("gu", "d")}
        for m, a in allocs.items(): json.dump(a, open(os.path.join(out, f"b{budget}_{m}.json"), "w"))
    json.dump(tokens, open(os.path.join(out, "calibration_tokens_per_expert.json"), "w")); log("allocation written to", out)


def repack(ckpt, out):
    """per layer L<i>.planes (one page-aligned slot per expert: U of gate/up/down, then H1 .. H6 of each; mode K reads the prefix) and
    L<i>.tab<K> (the table of mode K: fp16 [rows, 2^K], the mean of the children of the 8-bit table; K = 7: the 8-bit codes, lo, step)"""
    os.makedirs(out, exist_ok=True); refs = ar_tensors(ckpt); PAGE = 4096; pad = lambda n: (-n) % PAGE; PROJ = ("gate_proj", "up_proj", "down_proj")
    def tab(P, K):
        codes, lo, st = P["table_codes"], P["table_lo"].float(), P["table_step"].float()
        if K == 7: return torch.cat([codes.contiguous().view(-1), lo.half().view(-1).view(torch.uint8), st.half().view(-1).view(torch.uint8)])
        raw = lo + codes.float() * st; g = 1 << (7 - K); return (raw.view(raw.shape[0], 1 << K, g).sum(-1) / g).half().contiguous().view(-1).view(torch.uint8)
    for li in range(NL):
        if os.path.exists(os.path.join(out, f"L{li}.tab7")): continue
        P = {(e, pr): load(refs[f"model.layers.{li}.mlp.experts.{e}.{pr}"]) for e in range(E) for pr in PROJ}
        with open(os.path.join(out, f"L{li}.planes"), "wb") as f:
            for e in range(E):
                b = torch.cat([P[(e, pr)][f"plane{p}"].contiguous().view(-1) for p in range(7) for pr in PROJ]); f.write(b.numpy().tobytes() + bytes(pad(b.numel())))
        for K in range(3, 8):
            with open(os.path.join(out, f"L{li}.tab{K}"), "wb") as f:
                for e in range(E):
                    b = torch.cat([tab(P[(e, pr)], K) for pr in PROJ]); f.write(b.numpy().tobytes() + bytes(pad(b.numel())))
        log(f"layer {li} repacked")


def nonexpert(src, out):
    """every non-expert tensor in one file, with the config and tokenizer files"""
    import shutil
    from safetensors.torch import save_file
    if os.path.exists(os.path.join(out, "nonexpert.safetensors")): return
    os.makedirs(out, exist_ok=True); sd = {}
    for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        with safe_open(f, "pt") as fh: sd.update({k: fh.get_tensor(k) for k in fh.keys() if ".mlp.experts." not in k})
    save_file(sd, os.path.join(out, "nonexpert.safetensors"), metadata={"format": "pt"})
    for f in glob.glob(os.path.join(src, "*.json")) + glob.glob(os.path.join(src, "*.txt")):
        if "index" not in os.path.basename(f): shutil.copy(f, out)


def main():
    from evaluation.__main__ import checkpoint
    from baselines.common import original
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["allocation", "speed"])
    ap.add_argument("--base", help="local folder of the original checkpoint (default: download it; speed: it must be on the NVMe that is measured)")
    ap.add_argument("--ckpt", help="Any-Rate checkpoint (default: Qwen3-30B-A3B-Instruct-2507-AR2to8, or Hugging Face)")
    ap.add_argument("--out", default="moe_results")
    a = ap.parse_args(); src = original("qwen3-30b-a3b-instruct-2507", a.base); ckpt = a.ckpt or checkpoint("qwen3-30b-a3b-instruct-2507", "ar", "AR2to8", None)
    if a.task == "allocation":
        sd = os.path.join(a.out, "allocation", "stats"); stats(src, ckpt, sd); allocate(sd, os.path.join(a.out, "allocation"))
    else:
        ne, rp, sp = (os.path.join(a.out, "speed", d) for d in ("nonexpert", "repacked", "")); nonexpert(src, ne); repack(ckpt, rp)
        for fmt, K in [("bf16", None)] + [("ar", K) for K in range(3, 8)]:
            for c in CACHES:
                f = os.path.join(sp, f"{fmt if K is None else f'ar_K{K}'}_c{c}.json")
                if os.path.exists(f): continue
                cmd = [sys.executable, "-m", "moe.engine", "--fmt", fmt, "--cache_gb", str(c), "--out", f]
                cmd += ["--model", src] if K is None else ["--model", ne, "--ar_dir", rp, "--K", str(K)]
                log(" ".join(cmd)); subprocess.run(cmd, check=True)
        print("decoding speed (tokens/s); rows: format, columns: VRAM expert cache (GB)\n" + " " * 8 + "".join(f"{c:>8d}" for c in CACHES))
        for name in ["bf16"] + [f"ar_K{K}" for K in range(3, 8)]:
            print(f"{name:8s}" + "".join(f"{json.load(open(os.path.join(sp, f'{name}_c{c}.json')))['summary']['tps']:8.2f}" for c in CACHES))


if __name__ == "__main__":
    main()
