"""GEMV time tables of the paper from gemv_llama.json: per linear layer (us) and per decoded token (ms; 32 layers x q, k, v, o, gate, up, down).
usage: python tables.py  (writes tables.md)"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "gemv_llama.json")))["rows"]; L = {r["layer"]: r for r in rows}
N = {"q/o": 2, "k/v": 2, "gate/up": 2, "down": 1}; B = ["2", "2.5", "3", "3.5", "4"]; METH = ["AP", "AB", "MG", "MG+perm", "AR"]
cell = lambda v, f: "-" if v is None else f.format(v)
out = ["# GEMV time per linear layer (us), Llama-3.1-8B-Instruct, batch 1, RTX 5070 Ti", "",
       "| layer (rows x columns) | method | " + " | ".join(B) + " |", "|---|---|" + "---|" * len(B)]
for lay, r in L.items():
    out.append(f"| {lay} ({r['M']}x{r['K']}) | FP16 (cuBLAS) | {r['fp16_us']:.1f} |" + " |" * (len(B) - 1))
    for m in METH: out.append(f"| | {m} | " + " | ".join(cell(r["us"].get(f"{m} {b}"), "{:.1f}") for b in B) + " |")
out += ["", "# Linear-layer GEMV time per decoded token (ms)", "", "| method | " + " | ".join(B) + " |", "|---|" + "---|" * len(B)]
for m in METH:
    tok = [32 * sum(N[l] * L[l]["us"][f"{m} {b}"] for l in N) / 1000 if f"{m} {b}" in L["q/o"]["us"] else None for b in B]
    out.append(f"| {m} | " + " | ".join(cell(v, "{:.2f}") for v in tok) + " |")
out += ["", "MG+perm: MG with the act_order input permutation of its released execution path."]
ap = [abs(r["cupti_ap_us"][f"copy {b}"] - r["cupti_ap_us"][f"official {b}"]) / r["cupti_ap_us"][f"official {b}"] for r in rows for b in (3, 4)]
out += [f"AP kernel of the released AB code vs the released AP kernel, profiler time at 3 and 4 bits: max difference {100 * max(ap):.2f}%"]
open(os.path.join(HERE, "tables.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out[-8:]))
