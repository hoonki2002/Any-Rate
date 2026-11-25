"""Encoder experiments of the paper on Qwen2.5-3B. Each table encodes the model with the settings it compares (once: finished layers and
measurements are skipped, and settings shared by several tables are reused), records the relative proxy loss tr(E H E^T) / tr(W H W^T) of
every matrix and mode during encoding, measures the WikiText-2 perplexity of every mode, and prints the table.

  python -m experiments layout      layouts UUUU, UHHHHU, HHHHHHU, UUHHHH and UHHHHHH (Appendix E, Layout)
  python -m experiments ablation    scalar ladder, mean propagation, half-bit modes off (Section 5, Ablation)
  python -m experiments residual    proxy loss of the 4-8-bit modes (Appendix D, residual planes)
  python -m experiments search      candidate searches on 12 matrices, and full enumeration on the whole model (Appendix E, Candidate Search)
  python -m experiments beam        beam widths 16, 64 and 256 (Appendix E, Candidate Search)
  python -m experiments weights     mode weights (Appendix E, Mode Weights)
  python -m experiments custom --layout UUHHHH --weights 1-1-2-3-4 --beam 16 --K 16 --target mean    any other setting

A setting is the layout, the mode weights (five values for the modes 2, 2.5, 3, 3.5 and 4 bits; the modes a layout does not have are
dropped, e.g. 1-1-2-4 for UHHHHU; or one value per mode of the layout), the beam width B (0: rank all candidates), the number K of
candidates evaluated exactly (0 with B = 0: every candidate), and the target rule (per_mode, or mean: MatGPTQ-style mean propagation).
Results: experiment_results/<model>/<setting>/."""
import argparse, glob, json, math, os, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from any_rate.core.pack import encode_layer
from any_rate.format import decode_planes, decode_residual

DEFAULT = {"layout": "UHHHHHH", "weights": "1-1-2-3-4", "beam": 64, "K": 64, "target": "per_mode"}
BITS = [2, 2.5, 3, 3.5, 4]
S = lambda **kw: {**DEFAULT, **kw}
TABLES = {
    "layout": [S(layout="UUUU"), S(layout="UHHHHU"), S(layout="HHHHHHU"), S(layout="UUHHHH"), S()],
    "ablation": [S(layout="UUUU"), S(target="mean"), S(weights="1-0.02-2-0.02-4"), S()],
    "residual": [S()],
    "beam": [S(beam=16), S(), S(beam=256)],
    "weights": [S()] + [S(weights=w) for w in ("1-1-1-1-1", "1-1-2-2-2", "1-2-3-5-7", "1-1-1-1-4", "3-2-1-1-1", "4-1-1-1-1", "7-5-3-2-1")],
    "search": [S(layout="UHHHHU", beam=0), S(layout="UHHHHU")],
}
log = lambda *x: print(time.strftime("[%H:%M:%S]"), *x, flush=True)
gm = lambda v: math.exp(sum(map(math.log, v)) / len(v))


def modes(layout):
    """(planes read, bits) of every mode of a layout: U = 1 bit, H = 1/2 bit; the lowest mode reads 2 bits"""
    out = [(k, sum(1 if c == "U" else 0.5 for c in layout[:k])) for k in range(1, len(layout) + 1)]
    return [(k, b) for k, b in out if b >= 2]


def mode_weights(s):
    w = [float(x) for x in s["weights"].split("-")]; ms = modes(s["layout"])
    if len(w) == len(ms): return tuple(w)
    assert len(w) == 5, s["weights"]
    return tuple(dict(zip(BITS, w))[b] for _, b in ms)


def tag(s): return "_".join(f"{k}={v}" for k, v in s.items() if v != DEFAULT[k]) or "default"


def relative(W, Wq, H):
    E = W - Wq; return float(((E @ H) * E).sum() / ((W @ H) * W).sum())


def encode(src, s, d, layers):
    """encode the layers of setting s that are not in d yet; the proxy loss of every matrix and mode goes to d/proxy_<layer>.json"""
    os.makedirs(d, exist_ok=True); json.dump(s, open(os.path.join(d, "setting.json"), "w"))
    todo = [li for li in layers if not os.path.exists(os.path.join(d, f"layer_{li}.safetensors"))]
    if not todo: return
    from any_rate.models import decoder
    ms = modes(s["layout"]); residual = s == DEFAULT        # residual planes only for the setting of the paper

    def enc(W, H):
        t = encode_layer(W, H, s["layout"], ms[0][0], mode_weights(s), s["beam"], s["K"], s["target"], residual)
        g = {k: v.cuda() for k, v in t.items()}; p = {f"{b:g}": relative(W, decode_planes(g, s["layout"], k), H) for k, b in ms}
        if residual:
            w4 = decode_planes(g, s["layout"], len(s["layout"]))
            p.update({f"{4 + k}": relative(W, decode_residual(g, w4, k), H) for k in (1, 2, 3, 4)})
        t["_proxy"] = p; return t

    def save(li, store):
        json.dump({n: t.pop("_proxy") for n, t in store.items()}, open(os.path.join(d, f"proxy_{li}.json"), "w"), indent=1)
        save_file({f"{n}.ar.{k}": v.contiguous() for n, t in store.items() for k, v in t.items()}, os.path.join(d, f"layer_{li}.safetensors"))

    decoder.encode(src, enc, save, layers=todo, log=log)


def ppl(src, s, d, nlayers):
    """WikiText-2 perplexity of every mode of the encoded model (d/ppl.json); FP16 for s = None"""
    f = os.path.join(d, "ppl.json"); res = json.load(open(f)) if os.path.exists(f) else {}
    want = ["fp16"] if s is None else [f"{b:g}" for _, b in modes(s["layout"])]
    if all(k in res for k in want): return res
    if s is not None and len(glob.glob(os.path.join(d, "layer_*.safetensors"))) < nlayers: return res
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from evaluation.tasks import ppl as wikitext2
    tok = AutoTokenizer.from_pretrained(src); model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float16).cuda().eval()
    mods = dict(model.named_modules())
    for k, b in ([(None, "fp16")] if s is None else [(k, f"{b:g}") for k, b in modes(s["layout"])]):
        if b in res: continue
        if k is not None:
            for fn in glob.glob(os.path.join(d, "layer_*.safetensors")):
                with safe_open(fn, "pt", device="cuda") as fh:
                    t = {}
                    for key in fh.keys(): name, field = key.split(".ar."); t.setdefault(name, {})[field] = fh.get_tensor(key)
                for name, tt in t.items(): mods[name].weight.data = decode_planes(tt, s["layout"], k).half()
        res[b] = wikitext2(model, tok, False)[1]["wikitext2"]; json.dump(res, open(f, "w"), indent=1); log(f"{tag(s) if s else 'FP16'} {b}: PPL {res[b]:.2f}")
    return res


def proxies(d):
    return {(int(os.path.basename(f)[6:-5]), n): v for f in glob.glob(os.path.join(d, "proxy_*.json")) for n, v in json.load(open(f)).items()}


def bits_per_weight(d, layout):
    """index bits plus the table (8-bit codes, fp16 lo and step) per weight, from the stored tensors"""
    bits = n = 0
    for fn in glob.glob(os.path.join(d, "layer_*.safetensors")):
        with safe_open(fn, "pt") as fh:
            for key in fh.keys():
                field = key.split(".ar.")[1]; sh = fh.get_slice(key).get_shape(); size = math.prod(sh)
                if field.startswith("res_"): continue
                bits += size * (16 if field in ("table_lo", "table_step") else 8)
                if field == "plane0": n += sh[0] * sh[1] * (8 if layout[0] == "U" else 16)
    return bits / n if n else float("nan")


def row(cells, fmt): return " ".join(fmt.format(c) if isinstance(c, float) else f"{c:>7s}" for c in cells)


def search(src, root, device):
    """12 matrices (k, o, gate, down of layers 3, 18, 33), 128 random rows each, equal mode weights: proxy loss of each search"""
    from any_rate.core.ladder import encode_ladder, proxy
    mf = os.path.join(root, "search", "matrices.pt"); rf = os.path.join(root, "search", "runs.json"); os.makedirs(os.path.dirname(mf), exist_ok=True)
    if not os.path.exists(mf):
        from any_rate.models import decoder
        keep = {}; want = ("self_attn.k_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj")
        def save(li, store):
            for n, t in store.items():
                if n.split(f"layers.{li}.")[1] in want:
                    rows = torch.randperm(t["W"].shape[0], generator=torch.Generator().manual_seed(0))[:128]; keep[n] = {"W": t["W"][rows], "H": t["H"]}
        decoder.encode(src, lambda W, H: {"W": W.cpu(), "H": H.cpu()}, save, layers=[3, 18, 33], log=log); torch.save(keep, mf)
    M = torch.load(mf); runs = json.load(open(rf)) if os.path.exists(rf) else []
    jobs = [("UHHHHU", "all", 0, 0), ("UHHHHU", "full", 0, 64)] + [("UHHHHU", f"B={B}", B, 64) for B in (16, 32, 64, 128, 256)] + [("UHHHHHH", f"B={B}", B, 64) for B in (16, 64, 256)]
    for lay, name, B, K in jobs:
        for m, t in M.items():
            if any(r["matrix"] == m and r["layout"] == lay and r["search"] == name for r in runs): continue
            W, H = t["W"].to(device), t["H"].to(device); ms = modes(lay); t0 = time.time()
            _, _, rec, _ = encode_ladder(W, H, lay, ms[0][0], tuple(1.0 for _ in ms), 0.01, B, K)
            if device == "cuda": torch.cuda.synchronize()
            base = float(((W @ H) * W).sum())
            runs.append({"matrix": m, "layout": lay, "search": name, "sec": time.time() - t0, "rel": {f"{b:g}": proxy(W, rec[k], H) / base for k, b in ms}})
            json.dump(runs, open(rf, "w"), indent=1); log(f"search {lay} {name} {m}: {runs[-1]['sec']:.1f}s")
    return runs


def main():
    from any_rate.encode import HF_IDS
    from baselines.common import original
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", choices=list(TABLES) + ["custom"])
    ap.add_argument("--model", default="qwen2.5-3b", choices=["qwen2.5-3b", "llama-3.1-8b-instruct", "phi-4"])
    ap.add_argument("--base", help="local folder of the original checkpoint (default: download it)")
    ap.add_argument("--out", default="experiment_results")
    ap.add_argument("--layers", type=int, nargs="*", help="encode only these layers (a quick check; no perplexity)")
    ap.add_argument("--device", default="cpu", help="search: device of the candidate searches (the paper times them on the CPU)")
    for k, v in DEFAULT.items(): ap.add_argument(f"--{k}", type=type(v), default=v, help="custom: " + k)
    a = ap.parse_args()
    src = original(a.model, a.base); root = os.path.join(a.out, a.model); n = json.load(open(os.path.join(src, "config.json")))["num_hidden_layers"]
    layers = a.layers if a.layers is not None else list(range(n))
    settings = TABLES.get(a.table) or [{k: getattr(a, k) for k in DEFAULT}, S()]
    runs = search(src, root, a.device) if a.table in ("search", "beam") else None   # beam: the search time on the 12 matrices
    res = {}
    for s in settings:
        d = os.path.join(root, tag(s)); encode(src, s, d, layers)
        res[tag(s)] = {"proxy": proxies(d), "ppl": ppl(src, s, d, n) if a.layers is None else {}, "bw": bits_per_weight(d, s["layout"])}
    fp16 = ppl(src, None, os.path.join(root, "fp16"), n) if a.layers is None else {}
    show(a.table, settings, res, fp16, runs)


def show(table, settings, res, fp16, runs):
    head = lambda: print(" " * 26 + row([f"{b:g}" for b in BITS], "") + " | " + row([f"{b:g}" for b in BITS], ""))
    def ratio(t, ref, keys=None):
        P, R = res[t]["proxy"], res[ref]["proxy"]; ks = [k for k in (keys or R) if k in P and k in R]
        return [gm([P[k][f"{b:g}"] / R[k][f"{b:g}"] for k in ks]) if ks and f"{b:g}" in P[ks[0]] else "--" for b in BITS], len(ks)
    pp = lambda t: [res[t]["ppl"].get(f"{b:g}", "--") for b in BITS]
    if table in ("layout", "ablation", "custom"):
        print(f"{table}: bits/weight | proxy loss relative to UHHHHHH (geometric mean over matrices) | WikiText-2 PPL (FP16 {fp16.get('fp16', float('nan')):.2f})"); head()
        for s in settings:
            t = tag(s); r, n = ratio(t, "default"); print(f"{s['layout'] if table == 'layout' else t:18s} {res[t]['bw']:5.2f}  " + row(r, "{:7.2f}") + " | " + row(pp(t), "{:7.2f}") + f"   ({n} matrices)")
    elif table == "residual":
        P = res["default"]["proxy"]; print(f"relative proxy loss (%), mean over {len(P)} matrices")
        print(" ".join(f"{b} bits: {100 * sum(v[b] for v in P.values()) / len(P):.3f}" for b in ("4", "5", "6", "7", "8")))
    elif table == "beam":
        ref = {"beam=16": "B=16", "default": "B=64", "beam=256": "B=256"}; R = {r["search"]: [] for r in runs}
        print("beam width: proxy loss relative to B = 256 | WikiText-2 PPL | search time on the 12 matrices relative to B = 256"); head()
        for s in settings:
            t = tag(s); r, n = ratio(t, "beam=256"); tm = [x for x in runs if x["layout"] == "UHHHHHH"]
            sec = lambda name: sum(x["sec"] for x in tm if x["search"] == name)
            print(f"{ref[t]:18s}       " + row(r, "{:7.3f}") + " | " + row(pp(t), "{:7.2f}") + f"   {sec(ref[t]) / sec('B=256'):.2f}x")
    elif table == "weights":
        print("mode weights: proxy loss relative to 1-1-2-3-4 (layers 3, 7, ..., 35) | WikiText-2 PPL"); head()
        for s in settings:
            t = tag(s); keys = [k for k in res["default"]["proxy"] if k[0] % 4 == 3]; r, n = ratio(t, "default", keys)
            print(f"{s['weights']:18s}       " + row(r, "{:7.3f}") + " | " + row(pp(t), "{:7.2f}") + f"   ({n} matrices)")
    elif table == "search":
        U = [x for x in runs if x["layout"] == "UHHHHU"]; ms = ["2", "2.5", "3", "4"]; ref = {x["matrix"]: x for x in U if x["search"] == "all"}
        print(f"UHHHHU, {len(ref)} matrices: proxy loss 2 / 2.5 / 3 / 4 bits and time, relative to evaluating all candidates")
        for name in ["all", "full"] + [f"B={B}" for B in (16, 32, 64, 128, 256)]:
            X = {x["matrix"]: x for x in U if x["search"] == name}; ks = [k for k in ref if k in X]
            print(f"  {name:6s} " + " ".join(f"{gm([X[k]['rel'][m] / ref[k]['rel'][m] for k in ks]):.3f}" for m in ms) + f"  {gm([X[k]['sec'] / ref[k]['sec'] for k in ks]):.2f}x")
        for s, name in zip(settings, ("full enumeration, top-K", "prefix beam B=64")):
            print(f"  PPL {name:24s} " + " ".join(f"{res[tag(s)]['ppl'].get(m, float('nan')):.2f}" for m in ms))


if __name__ == "__main__":
    main()
