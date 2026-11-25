"""Bits per weight of three ways to support the 5-8-bit modes, Qwen2.5-3B, all 252 linear layers (paper: Method Details, residual planes).
The ladder and residual terms are checked byte for byte against the released checkpoint (checkpoint_bytes_qwen25_3b.json).   usage: python residual_cost.py"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SH = [(2048, 2048), (256, 2048), (256, 2048), (2048, 2048), (11008, 2048), (11008, 2048), (2048, 11008)]   # q k v o gate up down (rows x columns), 36 layers
N = 36 * sum(m * k for m, k in SH); ROWS = 36 * sum(m for m, _ in SH)
table = lambda b: ROWS * (8 * 2 ** b + 32) / 8          # bytes: 2^b 8-bit codes + fp16 lo and step per row
ladder = {"ar.plane0": N // 8, **{f"ar.plane{p}": N // 16 for p in range(1, 7)}, "ar.table_codes": ROWS * 128, "ar.table_lo": ROWS * 2, "ar.table_step": ROWS * 2}
residual = {"ar.res_planes": N // 2, "ar.res_lev": ROWS * 30 * 2}                    # four planes + 30 fp16 offsets per row
assert {**ladder, **residual} == json.load(open(os.path.join(HERE, "checkpoint_bytes_qwen25_3b.json")))["bytes"]
bpw = lambda nbytes: 8 * nbytes / N
rows = [("Ladder only (2-4 bits)", 7, 4.0, bpw(table(7)), 0.0), ("Ladder extended by four U planes (2-8 bits)", 11, 8.0, bpw(table(11)), 0.0),
        ("Ladder + residual (2-8 bits)", 7, 8.0, bpw(table(7)), bpw(residual["ar.res_lev"]))]
print("| configuration | b | index planes | shared table | residual offsets | total |\n|---|---|---|---|---|---|")
for name, b, idx, tab, off in rows: print(f"| {name} | {b} | {idx:.2f} | {tab:.2f} | {off:.2f} | {idx + tab + off:.2f} |")
