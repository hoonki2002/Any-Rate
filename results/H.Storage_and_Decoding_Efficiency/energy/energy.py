"""Estimated decoding energy per token of Llama-3.1-8B-Instruct, batch 1 (paper: energy figure, bits-read table, energy estimate appendix).
E = (b_eff * N + B_common) * e_DRAM + 2 * M / eta, with e_DRAM = 3.7 pJ/bit and eta = 88.36 TOPS/W. b_eff: bits read per linear-layer weight
(codes + the metadata the format reads at that mode); B_common: fp16 lm_head + KV cache of 1024 positions; M: MACs per token.
Accuracy: mean of the eight benchmarks from ../../A.Full_Benchmark_Results/Llama-3.1-8B-Instruct/results_table.csv.   usage: python energy.py  (writes energy.json)"""
import csv, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
E_DRAM, ETA, CTX = 3.7, 88.36, 1024
D, NL, FF, V, NKV, HD, G = 4096, 32, 14336, 128256, 8, 128, 128
SH = [(4096, 4096), (1024, 4096), (1024, 4096), (4096, 4096), (FF, 4096), (FF, 4096), (4096, FF)]     # q k v o gate up down (rows x columns)
N = NL * sum(M * K for M, K in SH); ROWS = NL * sum(M for M, _ in SH)
per_row, per_group = (lambda bits: bits * ROWS / N), (lambda bits: bits / G)
B_EFF = {"AP": lambda b: b + per_row(16 * 2 ** b),                    # b planes + a table of 2^b fp16 values per row
         "AB": lambda b: b + per_group(16 * (b + 1)),                 # b planes + b scales and one offset (fp16) per group
         "MG (2/3/4)": lambda b: b + per_group(32), "MG (3/4/8)": lambda b: b + per_group(32),   # b-bit prefix of the master + scale and zero per group
         "AR (ours)": lambda b: b + per_row(16 * 2 ** round(2 * b - 1) if b < 4 else 128 * 8 + 32),   # fp16 mode table; 4 bits: 8-bit table + lo, step
         "FP16": lambda b: 16}
COMMON = V * D * 16 + NL * 2 * NKV * HD * 16 * CTX
def mj(m, b, e_dram=E_DRAM, eta=ETA): return ((B_EFF[m](b) * N + COMMON) * e_dram + 2 * (N + V * D) / eta) * 1e-9
ACC = {(r["method"], float(r["bit"])): float(r["Avg8"]) for r in csv.DictReader(open(os.path.join(HERE, "..", "..", "A.Full_Benchmark_Results", "Llama-3.1-8B-Instruct", "results_table.csv")))}
FP = ACC[("FP16", 16.0)]
BASE = ["AP", "AB", "MG (2/3/4)", "MG (3/4/8)"]; AR = "AR (ours)"; TARGETS = [55, 75, 85, 90]
def cheapest(m, t, **kw):
    ok = [(mj(m, b, **kw), b) for (mm, b), a in ACC.items() if mm == m and b <= 4 and 100 * a / FP >= t]
    return min(ok) if ok else None
def savings(**kw): return {f"{t}%": {m: round(100 * (1 - cheapest(AR, t, **kw)[0] / cheapest(m, t, **kw)[0]), 2) for m in BASE} for t in TARGETS}
modes = {m: {f"{b:g}": {"b_eff": round(B_EFF[m](b), 3), "mJ_per_token": round(mj(m, b), 1), "accuracy_pct_of_fp16": round(100 * ACC[(m, float(b))] / FP, 2)}
             for (mm, b) in sorted(ACC) if mm == m and b <= 4} for m in BASE + [AR]}
s = savings(); s3 = savings(e_dram=3 * E_DRAM); s10 = savings(eta=ETA / 10)
shift = max(abs(x[t][m] - s[t][m]) for x in (s3, s10) for t in s for m in s[t])
out = {"constants": {"e_DRAM_pJ_per_bit": E_DRAM, "eta_TOPS_per_W": ETA, "kv_positions": CTX, "linear_weights": N, "rows": ROWS},
       "fp16_mJ_per_token": round(mj("FP16", 16), 1), "modes": modes,
       "cheapest_mode_at_target": {f"{t}%": {m: cheapest(m, t)[1] for m in BASE + [AR]} for t in TARGETS},
       "energy_saved_by_AR_pct": s,
       "saved_range_pct": [min(v for t in s.values() for v in t.values()), max(v for t in s.values() for v in t.values())],
       "max_change_of_savings_pts_with_3x_e_DRAM_or_eta_div_10": round(shift, 2),
       # compute share at the cheapest AR mode (2 bits): MACs only (as in E), and with one extra MAC per weight each for table lookup and decoding
       "compute_share_of_AR_2bit_pct": {"MACs only": round(100 * 2 * (N + V * D) / ETA * 1e-9 / mj(AR, 2), 2),
                                        "MACs + lookup + decoding": round(100 * 2 * (3 * N + V * D) / ETA * 1e-9 / mj(AR, 2), 2)}}
json.dump(out, open(os.path.join(HERE, "energy.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("fp16_mJ_per_token", "saved_range_pct", "max_change_of_savings_pts_with_3x_e_DRAM_or_eta_div_10", "compute_share_of_AR_2bit_pct")}))
print({m: {b: v["b_eff"] for b, v in d.items()} for m, d in modes.items()}); print({m: d["4"]["mJ_per_token"] for m, d in modes.items()})
