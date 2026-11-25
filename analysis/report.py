"""Every number of the Analysis section and the Analysis Details appendix, recomputed from data/ (Qwen2.5-3B, 246 matrices, first 256 rows,
C4 Hessians).  Per-matrix files in data/per_matrix/<kind>_<layer>_<proj>.json:
  nest_feedback_gain_mg   scalar (UU..) and coded (UHH..) ladders at 2/3/4 bits: single-mode and nested loss with feedback (full H) and without (diag H)
  mg_compare{,124,348}_all  MatGPTQ checkpoints (2/3/4 with weights 1:1:1 and 1:2:4, and the official 3/4/8) against single-mode GPTQ
  nest_meanprop_mg        nested loss with per-mode targets vs mean propagation, same encoder
  mg_weights_ppl.json     WikiText-2 PPL of the MatGPTQ (2/3/4) checkpoints with weights 1:1:1 and 1:2:4
  block_model             12 matrices: exact two-mode block model (UU -> UUU, UHH -> UHHH) on the encoder's own targets and tables
usage: python -m analysis.report [DATA]   (DATA: the output of python -m analysis, or results/F.Analysis_Details/data)"""
import glob, json, math, os, re, itertools
import numpy as np
from scipy.stats import spearmanr
import sys
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis_results"); PM = os.path.join(DATA, "per_matrix")
db = lambda a, b: 10 * math.log10(a / b); m2 = lambda x: np.round(np.mean(x, 0), 2)
keys = sorted(re.match(r"nest_feedback_gain_mg_(\d+)_(\w+)\.json", os.path.basename(f)).groups() for f in glob.glob(f"{PM}/nest_feedback_gain_mg_*.json"))
load = lambda kind, k: json.load(open(f"{PM}/{kind}_{k[0]}_{k[1]}.json"))
R = {k: load("nest_feedback_gain_mg", k)["results"] for k in keys}; UU = {k: R[k]["UU"] for k in keys}; UH = {k: R[k]["UHH"] for k in keys}
MG = {k: load("mg_compare_all", k) for k in keys}; MG4 = {k: load("mg_compare124_all", k) for k in keys}; M348 = {k: load("mg_compare348_all", k) for k in keys}
MP = {k: load("nest_meanprop_mg", k) for k in keys}
print(f"{len(keys)} matrices")

print("\n== Coded plane, single mode (2/3/4 bits)")
G = [[db(UU[k]["single_full"][i], UH[k]["single_full"][i]) for i in range(3)] for k in keys]
bit = lambda j, a: a[0] ^ ((j >> 2) & 1) * a[1] ^ ((j >> 1) & 1) * a[2] ^ (j & 1) * a[3]          # codeword bit of column j for message a
info = sum(len({tuple(bit(j, a) for j in s) for a in itertools.product((0, 1), repeat=4)}) == 16 for s in itertools.combinations(range(8), 4))
print("four-column subsets of [8,4,4] that are information sets:", info, "of 70")
cert = json.load(open(f"{DATA}/cert_fixed_table_r512_c2048.json")); gc = json.load(open(f"{DATA}/gauss_cols_32768_b256.json"))
print("Gaussian, fixed tables on fresh rows, gain vs Lloyd-Max (dB):", {b: round(v["gain_vs_LM"], 2) for b, v in cert.items()}, "| 4-bit in-sample, 32768 columns:", round(gc["4"]["gain_vs_LM"], 2))
print("Qwen single-mode gain (dB):", m2(G), "matrices won:", [sum(g[i] > 0 for g in G) for i in range(3)])
W = json.load(open(f"{DATA}/wnsm_all_mg.json")); kk = [f"{a}_{b}" for a, b in keys]
print("G(d) prediction: rank corr", [round(float(spearmanr([W[x]["G"] for x in kk], [W[x]["single"][i] for x in kk])[0]), 2) for i in range(3)],
      "measured - predicted", [round(float(np.mean([W[x]["single"][i] - W[x]["G"] for x in kk])), 2) for i in range(3)])
A = json.load(open(f"{DATA}/alloc_sort_pred.json")); print("2-bit gain, original vs sorted column order:", round(np.mean([A[x]["gain"] for x in kk]), 2), round(np.mean([A[x]["gain_sorted"] for x in kk]), 2))

print("\n== Feedback gain and nesting cost (dB)")
print("feedback gain:", m2([UU[k]["fb_gain_single"] for k in keys]))
T = json.load(open(f"{DATA}/trace_ratio_damp_mg.json")); P = [T[x]["1e-08"] for x in kk]; M = [UU[k]["fb_gain_single"][2] for k in keys]
print("4-bit prediction 10log10(sum H_jj / sum d_j): rank corr", round(spearmanr(P, M)[0], 2), "predicted - measured", round(float(np.mean(np.array(P) - np.array(M))), 2))
print("scalar ladder, no feedback:", m2([UU[k]["cost_diag"] for k in keys]), "4-bit IQR", np.round(np.percentile([UU[k]["cost_diag"][2] for k in keys], [25, 75]), 2))
print("scalar ladder, per-mode targets:", m2([UU[k]["cost_full"] for k in keys]))
print("MatGPTQ 1:1:1:", m2([MG[k]["mg_cost"] for k in keys]), "| 1:2:4:", m2([MG4[k]["mg_cost"] for k in keys]))
print("MatGPTQ 1:1:1 4-bit cost > feedback gain:", sum(MG[k]["mg_cost"][2] > UU[k]["fb_gain_single"][2] for k in keys))
d2 = [db(MG4[k]["mg_slice"][0], MG[k]["mg_slice"][0]) for k in keys]
print("1:2:4 vs 1:1:1, 2-bit layer loss:", round(float(np.mean(d2)), 2), "dB, higher on", sum(x > 0 for x in d2),
      "| mean relative loss 2-bit", round(np.mean([MG[k]["mg_slice"][0] for k in keys]), 3), "->", round(np.mean([MG4[k]["mg_slice"][0] for k in keys]), 3),
      "4-bit", round(np.mean([MG[k]["mg_slice"][2] for k in keys]), 3), "->", round(np.mean([MG4[k]["mg_slice"][2] for k in keys]), 3))
mw = json.load(open(f"{DATA}/mg_weights_ppl.json"))["checkpoints"] if os.path.exists(f"{DATA}/mg_weights_ppl.json") else {}
for name, p in mw.items(): print(f"{name}: WikiText-2 PPL", [p[b] for b in ("2", "3", "4")])
print("MatGPTQ {3,4,8}: cost", m2([M348[k]["mg_cost"] for k in keys]), "8-bit nested / 4-bit GPTQ single (median)", round(float(np.median([M348[k]["abs_vs_gptq4"][2] for k in keys])), 2))
for lad in ("UU", "UHH"):
    ex = lambda d, rule: sum((d["results"][rule][lad]["nested"][i] - d["single"][lad][i]) / d["single"][lad][i] for i in range(3))
    r = np.array([ex(MP[k], "meanprop") / ex(MP[k], "permode") for k in keys])
    worse = [sum(MP[k]["results"]["meanprop"][lad]["nested"][i] > MP[k]["results"]["permode"][lad]["nested"][i] for k in keys) for i in range(3)]
    print(f"mean propagation / per-mode targets, {lad} ladder: median {np.median(r):.2f} (10-90% {np.percentile(r, 10):.2f}-{np.percentile(r, 90):.2f}), larger on {int((r > 1).sum())}; per mode larger on {worse}")

print("\n== Coded vs scalar ladder")
saved = [[UU[k]["cost_full"][i] - UH[k]["cost_full"][i] for i in range(3)] for k in keys]
NG = [[db(UU[k]["nested_full"][i], UH[k]["nested_full"][i]) for i in range(3)] for k in keys]
def total_bits_gain(k):   # scalar nested loss interpolated (log) to the total bits of the coded mode; each mode reads its 8-bit table (2^planes entries + fp16 lo, step) per row
    c = 11008 if k[1] == "down_proj" else 2048; tb = lambda n: (8 * 2 ** n + 32) / c; Lu = np.log10(UU[k]["nested_full"]); xs = [b + tb(b) for b in (2, 3, 4)]; out = []
    for i, (b, n) in enumerate(((2, 3), (3, 5), (4, 7))):
        x = b + tb(n); j = 0 if x < xs[1] else 1; y = Lu[j] + (Lu[j + 1] - Lu[j]) * (x - xs[j]) / (xs[j + 1] - xs[j]); out.append(10 * (y - math.log10(UH[k]["nested_full"][i])))
    return out
print("single-mode gain", m2(G), "+ nesting cost saved", m2(saved), "= nested gain", m2(NG), "(won", [sum(g[i] > 0 for g in NG) for i in range(3)], ") | same total bits", m2([total_bits_gain(k) for k in keys]))
print("structural cost (no feedback): coded", m2([UH[k]["cost_diag"] for k in keys]), "scalar", m2([UU[k]["cost_diag"] for k in keys]))
lin = lambda r, i: 10 ** (r["cost_full"][i] / 10) - 10 ** (r["cost_diag"][i] / 10)
print("conflict cost ratio coded / scalar (median):", np.round(np.median([[lin(UH[k], i) / lin(UU[k], i) for i in range(3)] for k in keys], 0), 2))
BM = [json.load(open(f))["results"] for f in sorted(glob.glob(f"{PM}/block_model_*.json"))]
rng = lambda x: f"{min(x):.2f}-{max(x):.2f}"
if BM: print(f"block model, {len(BM)} matrices, coded / scalar: mean squared gap", rng([b["sigma_ratio"] for b in BM]), "coefficient K", rng([b["K_ratio"] for b in BM]),
      "conflict cost", rng([b["Dconf_ratio"] for b in BM]), "| K: scalar", rng([b["UU"]["K"] for b in BM]), "coded", rng([b["UHH"]["K"] for b in BM]))
print("gain over scalar single mode without feedback: feedback alone", m2([UU[k]["fb_gain_single"] for k in keys]),
      "coded alone", m2([[db(UU[k]["single_diag"][i], UH[k]["single_diag"][i]) for i in range(3)] for k in keys]),
      "both", m2([[db(UU[k]["single_diag"][i], UH[k]["single_full"][i]) for i in range(3)] for k in keys]))
pw = json.load(open(f"{DATA}/paperweights_nested_cost_mg.json"))
print("coded ladder nesting cost, half-bit modes off -> paper ladder (1-1-2-3-4):", m2([[db(v["nested_analysis"][i], v["single_full"][i]) for i in range(3)] for v in pw.values()]), "->",
      m2([[db(v["nested_paper"][i], v["single_full"][i]) for i in range(3)] for v in pw.values()]))
dp = json.load(open(f"{DATA}/damp_paperladder.json")); ch = {k: [v["cost_0.01"][i] - v["cost_1e-08"][i] for i in range(3)] for k, v in dp.items()}
print(f"damping 0.01 vs 1e-8, paper ladder, {len(dp)} matrices: all modes within 0.3 dB on {sum(max(map(abs, v)) < 0.3 for v in ch.values())}, largest change",
      max(((k, round(x, 2)) for k, v in ch.items() for x in v), key=lambda t: abs(t[1])),
      "| 4-bit cost at 1e-8: paper ladder", round(np.mean([v["cost_1e-08"][2] for v in dp.values()]), 2),
      "analysis ladder", round(np.mean([UH[tuple(k.split("_", 1))]["cost_full"][2] for k in dp]), 2),
      "scalar ladder", round(np.mean([UU[tuple(k.split("_", 1))]["cost_full"][2] for k in dp]), 2), "MatGPTQ 1:1:1", round(np.mean([MG[tuple(k.split("_", 1))]["mg_cost"][2] for k in dp]), 2))
pc = [v["cost_1e-08"] for v in dp.values()]; ac = [UH[tuple(k.split("_", 1))]["cost_full"] for k in dp]
print("paper ladder at 1e-8 vs analysis ladder: pays more at 2 bits on", sum(p[0] > a[0] for p, a in zip(pc, ac)), "| pays less at 3 / 4 bits on",
      sum(p[1] < a[1] for p, a in zip(pc, ac)), "/", sum(p[2] < a[2] for p, a in zip(pc, ac)), "of", len(pc))
