"""Decoding throughput table of the paper from tps_llama.json (tokens/s).   usage: python table.py  (writes table.md)"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, "tps_llama.json")))["tokens_per_s"]; B = ["2", "2.5", "3", "3.5", "4"]
out = ["# Decoding throughput (tokens/s), Llama-3.1-8B-Instruct, batch 1, RTX 5070 Ti", "", "| method | " + " | ".join(B) + " |", "|---|" + "---|" * len(B)]
for m in ("AP", "AB", "MG", "MG+perm", "AR"):
    out.append(f"| {m} | " + " | ".join(f"{T[f'{m} {b}']['median']:.2f}" if f"{m} {b}" in T else "-" for b in B) + " |")
out += ["", f"Without linear layers: {T['none 0']['median']:.2f} tokens/s ({1000 / T['none 0']['median']:.1f} ms per token)."]
open(os.path.join(HERE, "table.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
