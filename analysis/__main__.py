"""Layer-loss analysis of the paper (Section 4 and Appendix F) on Qwen2.5-3B: every number of the section and the appendix, from the
measurements below. Every step writes its files to --out and is skipped when they exist.

  python -m analysis

  hessians     input Hessians of every layer with the calibration of the paper (C4, MatGPTQ-style, 128 x 2048)
  feedback     246 matrices (first 256 rows): scalar (UU..) and coded (UHH..) ladders at 2/3/4 bits, single-mode and nested
               (weights 1-2-4 and 1-.02-2-.02-4), encoded with the full Hessian (error feedback) and with its diagonal (none)
  meanprop     the nested ladders with MatGPTQ-style mean propagation instead of per-mode targets
  blockmodel   12 matrices: two-mode ladders one plane apart (UU -> UUU and UHH -> UHHH, equal weights), solved exactly per block on the
               targets and tables of the encoder's last round: mean squared gap, conflict cost and its coefficient
  percol       per-column importances d_j of GPTQ and the trace-ratio prediction of the feedback gain at several dampings
  alloc        per-column split of the 2-bit coded gain, in the original column order and sorted by importance
  gain         the predicted gain G(d) of the coded plane (Hamming point set against the cubic lattice under weighted decoding)
  paper        nesting cost of the released encoding (UHHHHHH, 1-1-2-3-4) and its re-encoding at dampings 1e-8 and 0.01 (12 matrices)
  matgptq      MatGPTQ checkpoints (2/3/4 with weights 1:1:1 and 1:2:4, official 3/4/8) against single-mode GPTQ on MatGPTQ's grid;
               needs the checkpoints of baselines/matgptq (--recipe 234, --recipe 234 --weights 1 2 4, --recipe 348)
  gaussian     i.i.d. Gaussian rows: the coded singles against Lloyd-Max, in sample (32768 columns) and on fresh rows with fixed tables
  report       the numbers (report.py)

The analysis ladders use damping 1e-8 x mean diagonal, beam B = K = 64 and N_ref = 4. Six matrices (layer 1 up, layer 2 gate/up/down,
layer 3 gate/up) are left out: the table refit of their coded ladder is singular."""
import argparse, glob, itertools, json, math, os, re, runpy, sys, time
import numpy as np
import torch
from safetensors import safe_open
from any_rate.core.ladder import encode_ladder, hamming8

MODS = {"q_proj": ("self_attn", "attn"), "k_proj": ("self_attn", "attn"), "v_proj": ("self_attn", "attn"), "o_proj": ("self_attn", "o"),
        "gate_proj": ("mlp", "mlp"), "up_proj": ("mlp", "mlp"), "down_proj": ("mlp", "down")}
SKIP = {(1, "up_proj"), (2, "gate_proj"), (2, "up_proj"), (2, "down_proj"), (3, "gate_proj"), (3, "up_proj")}
ROWS, DEV = 256, "cuda"
STEPS = ["hessians", "feedback", "meanprop", "blockmodel", "percol", "alloc", "gain", "paper", "matgptq", "gaussian", "report"]
log = lambda *x: print(time.strftime("[%H:%M:%S]"), *x, flush=True)
dB = lambda a, b: 10 * math.log10(a / b)


class Data:
    def __init__(self, src, out, layers=None):
        self.src, self.out, self.hd, self.pm = src, out, os.path.join(out, "hessians"), os.path.join(out, "per_matrix")
        self.work = os.path.join(out, "work"); [os.makedirs(d, exist_ok=True) for d in (self.hd, self.pm, self.work)]
        files = glob.glob(os.path.join(src, "*.safetensors")); self.where = {k: f for f in files for k in safe_open(f, "pt").keys()}
        self.nl = json.load(open(os.path.join(src, "config.json")))["num_hidden_layers"]
        self.layers = list(range(self.nl)) if layers is None else layers
        self.keys = [(L, m) for L in self.layers for m in MODS if (L, m) not in SKIP]

    def W(self, L, mod, rows=ROWS):
        k = f"model.layers.{L}.{MODS[mod][0]}.{mod}.weight"
        with safe_open(self.where[k], "pt") as f: return f.get_tensor(k).float()[:rows]

    def H(self, L, mod):
        H = torch.load(os.path.join(self.hd, f"L{L}.{MODS[mod][1]}.pt"), map_location="cpu").float(); return H / H.diagonal().mean()

    def path(self, kind, L, mod): return os.path.join(self.pm, f"{kind}_{L}_{mod}.json")


def rel(W, Wh, H): return float(((W - Wh) @ H * (W - Wh)).sum() / ((W @ H) * W).sum())


UU_S = [("UU", 2, (1,), 2), ("UUU", 3, (1,), 3), ("UUUU", 4, (1,), 4)]; UU_N = ("UUUU", 2, (1, 2, 4), (2, 3, 4))
UH_S = [("UHH", 3, (1,), 3), ("UHHHHHH", 3, (.02, .02, 1, .02, .02), 5), ("UHHHHHH", 3, (.02, .02, .02, .02, 1), 7)]; UH_N = ("UHHHHHH", 3, (1, .02, 2, .02, 4), (3, 5, 7))


def hessians(D):
    todo = [L for L in D.layers if not all(os.path.exists(os.path.join(D.hd, f"L{L}.{g}.pt")) for g in ("attn", "o", "mlp", "down"))]
    if not todo: return
    from any_rate.models import decoder
    def save(li, store):
        for name, t in store.items():
            f = os.path.join(D.hd, f"L{li}.{MODS[name.split('.')[-1]][1]}.pt")
            if not os.path.exists(f): torch.save(t["H"], f)
    decoder.encode(D.src, lambda W, H: {"H": H.cpu()}, save, layers=todo, log=log)


def feedback(D):
    for L, mod in D.keys:
        if os.path.exists(D.path("nest_feedback_gain_mg", L, mod)): continue
        W, H = D.W(L, mod).to(DEV), D.H(L, mod).to(DEV); Hd = torch.diag(H.diagonal()); out = {}; t0 = time.time()
        for tag, S, (lay, first, w, picks) in (("UU", UU_S, UU_N), ("UHH", UH_S, UH_N)):
            r = {}
            for enc, He in (("diag", Hd), ("full", H)):
                r["single_" + enc] = [rel(W, encode_ladder(W, He, l, f, ww, 1e-8, 64, 64)[2][p], H) for l, f, ww, p in S]
                rec = encode_ladder(W, He, lay, first, w, 1e-8, 64, 64)[2]; r["nested_" + enc] = [rel(W, rec[p], H) for p in picks]
            r["fb_gain_single"] = [dB(a, b) for a, b in zip(r["single_diag"], r["single_full"])]
            r["fb_gain_nested"] = [dB(a, b) for a, b in zip(r["nested_diag"], r["nested_full"])]
            r["cost_diag"] = [dB(n, s) for n, s in zip(r["nested_diag"], r["single_diag"])]; r["cost_full"] = [dB(n, s) for n, s in zip(r["nested_full"], r["single_full"])]
            out[tag] = r
        out["nested_gain_UHH_vs_UU"] = {e: [dB(a, b) for a, b in zip(out["UU"]["nested_" + e], out["UHH"]["nested_" + e])] for e in ("diag", "full")}
        out["UHH_full_vs_UU_diag_nested"] = [dB(a, b) for a, b in zip(out["UU"]["nested_diag"], out["UHH"]["nested_full"])]
        json.dump(dict(rows=ROWS, layer=L, mod=mod, results=out), open(D.path("nest_feedback_gain_mg", L, mod), "w"), indent=1)
        log(f"feedback L{L} {mod} [{time.time() - t0:.0f}s]")


def meanprop(D):
    for L, mod in D.keys:
        if os.path.exists(D.path("nest_meanprop_mg", L, mod)): continue
        W, H = D.W(L, mod).to(DEV), D.H(L, mod).to(DEV); ref = json.load(open(D.path("nest_feedback_gain_mg", L, mod)))["results"]; res = {}
        for prop, target in (("permode", "per_mode"), ("meanprop", "mean")):
            res[prop] = {}
            for tag, (lay, first, w, picks) in (("UU", UU_N), ("UHH", UH_N)):
                rec = encode_ladder(W, H, lay, first, w, 1e-8, 64, 64, target)[2]; nl = [rel(W, rec[p], H) for p in picks]
                res[prop][tag] = dict(nested=nl, cost=[dB(n, s) for n, s in zip(nl, ref[tag]["single_full"])])
        json.dump(dict(layer=L, mod=mod, rows=ROWS, single=dict(UU=ref["UU"]["single_full"], UHH=ref["UHH"]["single_full"]), results=res),
                  open(D.path("nest_meanprop_mg", L, mod), "w"), indent=1); log(f"meanprop L{L} {mod}")


BLOCK_KEYS = [(L, m) for L in (0, 17, 35) for m in ("q_proj", "o_proj", "gate_proj", "down_proj")]


def _record(W, H, lay, first):
    """Encode a two-mode ladder (equal weights) and record, for every block of the last round, the targets of both modes, the block
    weights d_j = 1 / R_jj^2 and the two mode tables that the block was encoded with."""
    import any_rate.core.ladder as lad
    rec, orig = [], lad.block_step
    def spy(ts, Ub, luts, *a):
        rec.append((ts[0].clone(), ts[-1].clone(), 1.0 / Ub.diagonal() ** 2, luts[0], luts[-1])); return orig(ts, Ub, luts, *a)
    lad.block_step = spy
    try: encode_ladder(W, H, lay, first, (1, 1), 1e-8, 64, 64)
    finally: lad.block_step = orig
    last = rec[-(W.shape[1] // 8):]
    tc, tf = (torch.stack([r[i] for r in last], 1) for i in (0, 1)); return tc, tf, torch.stack([r[2] for r in last]), last[-1][3], last[-1][4]


def _exact_scalar(tc, tf, d, lc, lf, wc=0.5, wf=0.5):
    """UU (4 levels) -> UUU (8 levels), separable per column: child j has parent j >> 1. Returns the summed excess over the single modes
    with the two targets (g) and with both modes at t_c or at t_f (g = 0)."""
    par = torch.arange(8, device=DEV) >> 1; vf = lf[:, None, None, :]; vc = lc[:, par][:, None, None, :]; vco = lc[:, None, None, :]; dd = d[None, :, :, None]
    def J(ta, tb):
        ec = (ta[..., None] - vc) ** 2 * dd; ef = (tb[..., None] - vf) ** 2 * dd
        return float(((wc * ec + wf * ef).min(-1).values - wc * ((ta[..., None] - vco) ** 2 * dd).min(-1).values - wf * ef.min(-1).values).sum())
    return J(tc, tf), J(tc, tc), J(tf, tf)


def _exact_coded(tc, tf, d, lc, lf, wc=0.5, wf=0.5, chunk=16):
    """UHH (8 levels) -> UHHH (16 levels): every one of the 2^8 * 16^3 index blocks, index = U << 3 | h1 << 2 | h2 << 1 | h3 per column,
    parent = index >> 1. Same returns as _exact_scalar."""
    R, B, _ = tc.shape; h = (hamming8().to(DEV) < 0).long(); U = torch.tensor(list(itertools.product([0, 1], repeat=8)), device=DEV)
    I3 = ((h[:, None, None] << 2) | (h[None, :, None] << 1) | h[None, None]).reshape(-1, 8); Ifull = ((U[:, None] << 3) | I3[None]).reshape(-1, 8)
    Ipar = Ifull >> 1; Ic = ((U[:, None] << 2) | ((h[:, None] << 1) | h[None]).reshape(-1, 8)[None]).reshape(-1, 8); Jg = J0c = J0f = 0.0
    for r0 in range(0, R, chunk):
        rows = slice(r0, min(R, r0 + chunk)); vf, vc, vco = lf[rows][:, Ifull], lc[rows][:, Ipar], lc[rows][:, Ic]
        for b in range(B):
            dd = d[b][None, None]; Tc = tc[rows, b][:, None]; Tf = tf[rows, b][:, None]
            ec = ((Tc - vc) ** 2 * dd).sum(-1); ef = ((Tf - vf) ** 2 * dd).sum(-1); ef_c = ((Tc - vf) ** 2 * dd).sum(-1); ec_f = ((Tf - vc) ** 2 * dd).sum(-1)
            Sc = ((Tc - vco) ** 2 * dd).sum(-1).min(1).values; Sf = ef.min(1).values; Sc_f = ((Tf - vco) ** 2 * dd).sum(-1).min(1).values
            Jg += float(((wc * ec + wf * ef).min(1).values - wc * Sc - wf * Sf).sum())
            J0c += float(((wc * ec + wf * ef_c).min(1).values - wc * Sc - wf * ef_c.min(1).values).sum())
            J0f += float(((wc * ec_f + wf * ef).min(1).values - wc * Sc_f - wf * Sf).sum())
    return Jg, J0c, J0f


def blockmodel(D):
    for L, mod in [k for k in BLOCK_KEYS if k in D.keys]:
        if os.path.exists(D.path("block_model", L, mod)): continue
        W, H = D.W(L, mod).to(DEV), D.H(L, mod).to(DEV); out = {}; t0 = time.time()
        for tag, lay, first, fn in (("UU", "UUU", 2, _exact_scalar), ("UHH", "UHHH", 3, _exact_coded)):
            tc, tf, d, lc, lf = _record(W, H, lay, first); wsum = float(d.sum()) * tc.shape[0]
            sig = float((((tf - tc) ** 2) * d[None]).sum() / wsum); Jg, J0c, J0f = (x / wsum for x in fn(tc, tf, d, lc, lf)); D0 = (J0c + J0f) / 2
            out[tag] = dict(sigma_g2=sig, D_g=Jg, D_0=D0, D_conf=Jg - D0, K=(Jg - D0) / sig, D0c=J0c, D0f=J0f)
        out.update(sigma_ratio=out["UHH"]["sigma_g2"] / out["UU"]["sigma_g2"], K_ratio=out["UHH"]["K"] / out["UU"]["K"], Dconf_ratio=out["UHH"]["D_conf"] / out["UU"]["D_conf"])
        json.dump(dict(layer=L, mod=mod, rows=ROWS, results=out), open(D.path("block_model", L, mod), "w"), indent=1); log(f"blockmodel L{L} {mod} [{time.time() - t0:.0f}s]")


def chol_U(H, damp=1e-8):
    n = H.shape[0]; return torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H.double() + damp * torch.eye(n, device=H.device, dtype=torch.float64))), upper=True)


def percol(D):
    f = os.path.join(D.out, "trace_ratio_damp_mg.json")
    if os.path.exists(f): return
    out = {}
    for L, mod in D.keys:
        H = D.H(L, mod).double().to(DEV); rec = {}
        for damp in (1e-8, 1e-6, 1e-4, 1e-2):
            try: U = chol_U(H, damp)
            except Exception: continue
            d = 1.0 / U.diagonal() ** 2; rec[str(damp)] = 10 * math.log10(float(H.diagonal().sum() / d.sum()))
            if damp == 1e-8: np.savez(os.path.join(D.work, f"percol_rho_mg_{L}_{mod}.npz"), d=d.cpu().numpy())
        out[f"{L}_{mod}"] = rec
    json.dump(out, open(f, "w"), indent=1); log("percol done")


def alloc(D):
    for L, mod in D.keys:
        for sort in ("", "_sortd"):
            f = os.path.join(D.work, f"alloc_percol_L{L}_{mod}{sort}.npz")
            if os.path.exists(f): continue
            W, H = D.W(L, mod).to(DEV), D.H(L, mod).to(DEV); n = H.shape[0]; d_nat = (1.0 / chol_U(H).diagonal() ** 2).cpu().numpy()
            if sort: perm = torch.as_tensor(np.argsort(-d_nat), device=DEV); W = W[:, perm].contiguous(); H = H[perm][:, perm].contiguous()
            U = chol_U(H); Uinv = torch.linalg.solve_triangular(U, torch.eye(n, device=DEV, dtype=torch.float64), upper=True); den = float(((W @ H) * W).sum())
            cols = {}
            for tag, lay, first in (("UU", "UU", 2), ("UHH", "UHH", 3)):
                Wh = encode_ladder(W, H, lay, first, (1,), 1e-8, 64, 64)[2][first]; cols[tag] = (((W - Wh).double() @ Uinv) ** 2).sum(0).cpu().numpy() / den
            np.savez(f, d=(1.0 / U.diagonal() ** 2).cpu().numpy(), cu=cols["UU"], ch=cols["UHH"]); log(f"alloc L{L} {mod}{sort}")


def _G_setup():
    pts = np.array([[(i >> b) & 1 for b in range(3)] for i in range(8)]); Gm = np.vstack([np.ones(8, int), pts.T]) % 2
    C = np.array(sorted({tuple((m @ Gm) % 2) for m in itertools.product([0, 1], repeat=4)}))
    reps = torch.tensor([c1 + 2 * c2 for c1 in C for c2 in C], dtype=torch.float32, device=DEV)
    X = torch.rand((4000, 8), generator=torch.Generator(device=DEV).manual_seed(3), device=DEV) * 4
    Y = X[:, None, :] - reps[None, :, :]; Y = Y - 4 * torch.round(Y / 4); return Y ** 2


def G_of(d, Y2):
    """high-resolution weighted-decoding gain (dB) of the Hamming point set over the cubic lattice for importances d (blocks of 8)"""
    blk = d[: len(d) // 8 * 8].reshape(-1, 8); out = []
    for i in range(0, len(blk), 64):
        out.append(torch.einsum("nrj,bj->bnr", Y2, torch.tensor(blk[i:i + 64], dtype=torch.float32, device=DEV)).min(-1).values.mean(1) / 8)
    return 10 * math.log10((blk.mean(1) / 3).sum() / torch.cat(out).cpu().numpy().sum())


def gain(D):
    Y2 = _G_setup(); f = os.path.join(D.out, "wnsm_all_mg.json")
    if not os.path.exists(f):
        res = {}
        for L, mod in D.keys:
            d = np.load(os.path.join(D.work, f"percol_rho_mg_{L}_{mod}.npz"))["d"]; blk = d[: len(d) // 8 * 8].reshape(-1, 8)
            meas = json.load(open(D.path("nest_feedback_gain_mg", L, mod)))["results"]
            res[f"{L}_{mod}"] = dict(G=G_of(d, Y2), amgm_med=float(np.median(10 * np.log10(blk.mean(1) / np.exp(np.log(blk).mean(1))))),
                                     single=[dB(meas["UU"]["single_full"][b], meas["UHH"]["single_full"][b]) for b in range(3)],
                                     nested=[dB(meas["UU"]["nested_full"][b], meas["UHH"]["nested_full"][b]) for b in range(3)])
        json.dump(res, open(f, "w"), indent=1)
    f = os.path.join(D.out, "alloc_sort_pred.json")
    if not os.path.exists(f):
        def g(z): return 10 * math.log10(z["cu"].sum() / z["ch"].sum())
        def ceil(d): B = d.reshape(-1, 8); return 10 * np.log10(B.mean(1).sum() / np.exp(np.log(B).mean(1)).sum())
        def by_rank(z):
            d, cu, ch = z["d"], z["cu"], z["ch"]; nb = len(d) // 8; r = np.argsort(np.argsort(-d.reshape(nb, 8), 1), 1).reshape(-1)
            return [10 * math.log10(cu[r == k].sum() / ch[r == k].sum()) for k in range(8)]
        res = {}
        for L, mod in D.keys:
            a, b = (np.load(os.path.join(D.work, f"alloc_percol_L{L}_{mod}{s}.npz")) for s in ("", "_sortd"))
            r = dict(gain=g(a), gain_sorted=g(b), G=G_of(a["d"], Y2), G_sorted=G_of(b["d"], Y2), ceil=float(ceil(a["d"])), ceil_sorted=float(ceil(b["d"])), rank=by_rank(a), rank_sorted=by_rank(b))
            r["drop"] = r["gain"] - r["gain_sorted"]; r["pred_drop"] = r["G"] - r["G_sorted"]; r["ceil_drop"] = r["ceil"] - r["ceil_sorted"]; res[f"{L}_{mod}"] = r
        json.dump(res, open(f, "w"), indent=1)
    log("gain done")


def paper(D, ar_ckpt):
    from any_rate.format import decode_planes
    f = os.path.join(D.out, "paperweights_nested_cost_mg.json")
    if not os.path.exists(f):
        files = glob.glob(os.path.join(ar_ckpt, "*.safetensors")); where = {k: fn for fn in files for k in safe_open(fn, "pt").keys() if ".ar." in k}; out = {}
        for L, mod in D.keys:
            name = f"model.layers.{L}.{MODS[mod][0]}.{mod}"; t = {}
            for k in [k for k in where if k.startswith(name + ".ar.") and ".ar.res_" not in k]:
                with safe_open(where[k], "pt", device=DEV) as fh: t[k.split(".ar.")[1]] = fh.get_tensor(k)
            W, H = D.W(L, mod).to(DEV), D.H(L, mod).to(DEV); r = {k: rel(W, decode_planes(t, "UHHHHHH", k)[:ROWS], H) for k in (3, 4, 5, 6, 7)}
            ref = json.load(open(D.path("nest_feedback_gain_mg", L, mod)))["results"]["UHH"]; nested = [r[3], r[5], r[7]]
            out[f"{L}_{mod}"] = dict(nested_paper=nested, half=[r[4], r[6]], single_full=ref["single_full"], nested_analysis=ref["nested_full"],
                                    cost_paper=[dB(a, b) for a, b in zip(nested, ref["single_full"])], cost_analysis=ref["cost_full"])
        json.dump(out, open(f, "w"), indent=1)
    f2 = os.path.join(D.out, "damp_paperladder.json"); pw = json.load(open(f)); out = json.load(open(f2)) if os.path.exists(f2) else {}
    for L in (4, 18, 33):
        for mod in ("q_proj", "o_proj", "gate_proj", "down_proj"):
            if f"{L}_{mod}" in out: continue
            W, H = D.W(L, mod).to(DEV), D.H(L, mod).to(DEV); single = json.load(open(D.path("nest_feedback_gain_mg", L, mod)))["results"]["UHH"]["single_full"]
            r = {"cost_stored": pw[f"{L}_{mod}"]["cost_paper"]}
            for damp in (1e-8, 0.01):
                rec = encode_ladder(W, H, "UHHHHHH", 3, (1, 1, 2, 3, 4), damp, 64, 64)[2]; r[f"cost_{damp:g}"] = [dB(rel(W, rec[p], H), s) for p, s in zip((3, 5, 7), single)]
            out[f"{L}_{mod}"] = r; json.dump(out, open(f2, "w"), indent=1); log(f"paper ladder dampings L{L} {mod}")


def gptq_uniform(W, H, bits, group=128, damp=0.01, block=128):
    """GPTQ on MatGPTQ's grid: asymmetric uniform per group of columns, with a diag(H)-weighted clip search"""
    W = W.clone(); n = W.shape[1]; Hd = H.clone(); Hd += torch.diag(Hd).mean() * damp * torch.eye(n, dtype=W.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hd)), upper=True); Q = torch.zeros_like(W); maxq = 2 ** bits - 1; scale = zero = None
    for i1 in range(0, n, block):
        i2 = min(i1 + block, n); W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            if (i1 + i) % group == 0:
                g = W[:, i1 + i:i1 + i + group]; lo0 = g.min(1).values.clamp(max=0); hi0 = g.max(1).values.clamp(min=0); hw = torch.diag(H)[i1 + i:i1 + i + group][None]; best = None
                for p in torch.linspace(1.0, 0.2, 41):
                    lo, hi = lo0 * p, hi0 * p; s = ((hi - lo) / maxq).clamp_min(1e-8); z = torch.round(-lo / s)
                    q = (torch.clamp(torch.round(g / s[:, None]) + z[:, None], 0, maxq) - z[:, None]) * s[:, None]; e = (((g - q) ** 2) * hw).sum(1)
                    if best is None: best = e.clone(); scale, zero = s.clone(), z.clone()
                    else: m = e < best; best[m] = e[m]; scale[m] = s[m]; zero[m] = z[m]
            w = W1[:, i]; q = (torch.clamp(torch.round(w / scale) + zero, 0, maxq) - zero) * scale
            Q1[:, i] = q; err = (w - q) / Hinv1[i, i]; W1[:, i:] -= err[:, None] * Hinv1[i, i:][None]; E1[:, i] = err
        Q[:, i1:i2] = Q1; W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q


def mg_slice(ckpt, name, b, master):
    """MatGPTQ's slice rule on the stored master codes, columns un-permuted"""
    d = torch.load(glob.glob(os.path.join(ckpt, "**", name, "data.pt"), recursive=True)[0], map_location="cpu")
    q, scale, zero = d["qweight"].long(), d["scale"].double(), d["zero"].double(); perm = d["perm"].argsort()
    qs = (q >> (master - b)) << (master - b); off = (2 ** (master - b) - 1) / 2; G = scale.shape[1]
    return (scale.view(q.shape[0], G, 1) * (qs.view(q.shape[0], G, -1) - (zero.view(q.shape[0], G, 1) - off))).view_as(q)[:, perm][:ROWS]


def _mg_job(args):
    src, out, kind, ckpt, L, mod = args; torch.set_num_threads(4); D = Data(src, out); f = D.path(kind, L, mod)
    if os.path.exists(f): return
    W, H = D.W(L, mod).double(), D.H(L, mod).double(); name = f"model.layers.{L}.{MODS[mod][0]}.{mod}"
    bits, master = ((3, 4, 8), 8) if kind == "mg_compare348_all" else ((2, 3, 4), 4)
    mg = [rel(W, mg_slice(ckpt, name, b, master), H) for b in bits]; gs = [rel(W, gptq_uniform(W, H, b), H) for b in bits]
    r = dict(layer=L, mod=mod, rows=ROWS, gptq_single=gs, mg_slice=mg, mg_cost=[dB(a, b) for a, b in zip(mg, gs)])
    if kind == "mg_compare348_all": r.update(bits=bits, abs_vs_gptq4=[x / gs[1] for x in mg])
    json.dump(r, open(f, "w"), indent=1)


def matgptq(D, ckpts, nproc):
    import multiprocessing as mp
    jobs = [(D.src, D.out, kind, c, L, mod) for kind, c in ckpts.items() for L, mod in D.keys if not os.path.exists(D.path(kind, L, mod))]
    with mp.get_context("spawn").Pool(nproc) as p:
        for i, _ in enumerate(p.imap_unordered(_mg_job, jobs)): log(f"matgptq {i + 1}/{len(jobs)}")


def gaussian(D):
    f = os.path.join(D.out, "gauss_cols_32768_b256.json"); LM = {2: 0.117482, 3: 0.034548, 4: 0.009501}
    S = {2: ("UHH", (1,), 3), 3: ("UHHHH", (.02, .02, 1), 5), 4: ("UHHHHHH", (.02, .02, .02, .02, 1), 7)}
    if not os.path.exists(f):
        R, N = 2048, 32768; W = torch.randn((R, N), generator=torch.Generator(device=DEV).manual_seed(0), device=DEV); H = torch.eye(N, device=DEV); out = {}
        for b, (lay, w, p) in S.items():
            e = (W - encode_ladder(W, H, lay, 3, w, 1e-8, 256, 64)[2][p]) ** 2; m = e.mean().item()
            out[str(b)] = dict(mse=m, se=e.mean(1).std().item() / math.sqrt(R), gain_vs_LM=dB(LM[b], m)); log(f"gaussian in sample {b} bits")
        json.dump(out, open(f, "w"), indent=1)
    f = os.path.join(D.out, "cert_fixed_table_r512_c2048.json")
    if not os.path.exists(f):
        R, N = 512, 2048; g = torch.Generator(device=DEV).manual_seed(11); W1 = torch.randn((R, N), generator=g, device=DEV); W2 = torch.randn((R, N), generator=g, device=DEV)
        H = torch.eye(N, device=DEV); C = hamming8().to(DEV); pat = ((C < 0).long() * (1 << torch.arange(8, device=DEV))).sum(1); out = {}
        bits_of = lambda p: ((p[:, None] >> torch.arange(8, device=DEV)) & 1)
        for b, nH in ((2, 2), (3, 4)):
            lay, w, p = S[b]; _, (codes, lo, step), rec, _ = encode_ladder(W1, H, lay, 3, w, 1e-8, 64, 64)
            T = lo.float() + codes.float() * step.float(); nb = N // 8; L = 2 ** nH                         # index = u * 2^nH + coded bits
            combos = torch.tensor(list(itertools.product(range(16), repeat=nH)), device=DEV)
            rest = sum(bits_of(pat[combos[:, q]]) << (nH - 1 - q) for q in range(nH)); X = W2.reshape(R, nb, 8); tot = 0.0
            for r0 in range(0, R, 32):                     # exact search with the tables fixed: best U bit per coded index, all coded tuples
                x = X[r0:r0 + 32]; e = (x[..., None] - T[r0:r0 + 32][:, None, None, :]) ** 2; E = torch.minimum(e[..., :L], e[..., L:])
                cost = torch.zeros((e.shape[0], nb, rest.shape[0]), device=DEV)
                for j in range(8): cost += E[:, :, j, :].gather(-1, rest[None, None, :, j].expand(e.shape[0], nb, -1))
                tot += cost.min(-1).values.sum().item()
            mse = tot / (R * N); out[str(b)] = dict(mse=mse, gain_vs_LM=dB(LM[b], mse), in_sample=((W1 - rec[p]) ** 2).mean().item()); log(f"gaussian fixed tables {b} bits")
        json.dump(out, open(f, "w"), indent=1)


def main():
    from baselines.common import original
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="*", choices=STEPS, default=STEPS)
    ap.add_argument("--base", help="local folder of Qwen2.5-3B (default: download it)")
    ap.add_argument("--ar", help="the Any-Rate checkpoint of Qwen2.5-3B (default: Qwen2.5-3B-AR2to8, or Hugging Face)")
    ap.add_argument("--mg", nargs=3, metavar=("MG234", "MG234_W124", "MG348"), default=["Qwen2.5-3B-MG234", "Qwen2.5-3B-MG234-w124", "Qwen2.5-3B-MG348"])
    ap.add_argument("--nproc", type=int, default=8, help="matgptq: CPU processes")
    ap.add_argument("--layers", type=int, nargs="*", help="only these layers (a quick check)")
    ap.add_argument("--out", default="analysis_results")
    a = ap.parse_args(); D = Data(original("qwen2.5-3b", a.base), a.out, a.layers)
    for s in a.steps:
        log("step", s)
        if s == "hessians": hessians(D)
        elif s == "feedback": feedback(D)
        elif s == "meanprop": meanprop(D)
        elif s == "blockmodel": blockmodel(D)
        elif s == "percol": percol(D)
        elif s == "alloc": alloc(D)
        elif s == "gain": gain(D)
        elif s == "paper":
            from evaluation.__main__ import checkpoint
            paper(D, a.ar or checkpoint("qwen2.5-3b", "ar", "AR2to8", None))
        elif s == "matgptq":
            if not all(os.path.isdir(c) for c in a.mg): log("matgptq skipped: encode the MatGPTQ checkpoints first (see the docstring)"); continue
            matgptq(D, dict(zip(("mg_compare_all", "mg_compare124_all", "mg_compare348_all"), a.mg)), a.nproc)
        elif s == "gaussian": gaussian(D)
        elif s == "report":
            sys.argv = ["report", a.out]; runpy.run_module("analysis.report", run_name="__main__")


if __name__ == "__main__":
    main()
