"""Mixture-of-experts appendix (Qwen3-30B-A3B-Instruct-2507, experts encoded): the accuracy table and the numbers of the text.  accuracy/<run>/<bits>/<task>/: BF16, AR_uniform, AR_gate_mass and AR_reverse_gate_mass (per-expert modes from allocation/),
AR_attention_only and AR_experts_and_attention (other parts encoded), MG_2-3-4 (experts, PPL only).  usage: python moe_results.py"""
import glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ACC = os.path.join(HERE, "accuracy")
T5 = ("arc_challenge", "arc_easy", "hellaswag", "piqa_baber", "winogrande")
def summ(run, bits, task):
    f = glob.glob(f"{ACC}/{run}/{bits}/{task}/*_summary.json"); return json.load(open(f[0])) if f else None
def row(run, bits):
    p = summ(run, bits, "ppl")["wikitext2"]; c = summ(run, bits, "csr")
    if c is None: return {"ppl": p}
    s = [100 * c[t] for t in T5] + [100 * summ(run, bits, "gsm8k")["flexible"], 100 * summ(run, bits, "triviaqa")["em"], 100 * summ(run, bits, "nqopen")["em"]]
    return {"ppl": p, "scores": s, "avg": sum(s) / 8}
B = ["2", "2.5", "3", "3.5", "4"]; MB = ["2.5", "3", "3.5"]
R = {"BF16": row("BF16", "bf16"), "uniform": {b: row("AR_uniform", f"{b}bit") for b in B}, "gate_mass": {b: row("AR_gate_mass", f"{b}bit") for b in MB},
     "reverse": {b: row("AR_reverse_gate_mass", f"{b}bit") for b in MB}, "attention_only": {b: row("AR_attention_only", f"{b}bit") for b in B},
     "experts_and_attention": {b: row("AR_experts_and_attention", f"{b}bit") for b in B}, "MG_2-3-4": {b: row("MG_2-3-4", f"{b}bit") for b in ("2", "3", "4")}}

# accuracy table: PPL, eight benchmarks and their mean, uniform and gate-mass allocation per mean bit width
cols = ["PPL", "ARC-c", "ARC-e", "HellaSwag", "PIQA", "WinoGrande", "GSM8K", "TriviaQA", "NQ-open", "Avg."]
print("bits  allocation " + " ".join(f"{c:>10s}" for c in cols))
show = lambda b, name, r: print(f"{b:5s} {name:10s} " + " ".join(f"{v:10.2f}" for v in [r["ppl"]] + r["scores"] + [r["avg"]]))
for b in B:
    show(b, "uniform", R["uniform"][b])
    if b in MB: show(b, "gate mass", R["gate_mass"][b])
show("BF16", "-", R["BF16"])
print()

# numbers of the text
bf = R["BF16"]; U = R["uniform"]
cnt = [n for l in json.load(open(os.path.join(HERE, "allocation", "calibration_tokens_per_expert.json"))).values() for n in l.values()]
print(f"calibration tokens per expert: median {np.median(cnt):.0f}, fewer than 512: {sum(n < 512 for n in cnt)} of {len(cnt)}, none: {sum(n == 0 for n in cnt)}")
print("mean accuracy, % of BF16:", {b: round(100 * U[b]["avg"] / bf["avg"], 2) for b in B})
drop = [round(x - y, 2) for x, y in zip(bf["scores"], U["2"]["scores"])]; print("2-bit drop per task (arc_c arc_e hs piqa wino gsm8k tqa nq):", drop)
print("PPL at 2 bits: experts", round(U["2"]["ppl"], 2), "(+", round(U["2"]["ppl"] - bf["ppl"], 2), ") attention only", round(R["attention_only"]["2"]["ppl"], 2), "both", round(R["experts_and_attention"]["2"]["ppl"], 2))
print("AR vs MG (2/3/4) PPL:", {b: (round(U[b]["ppl"], 2), round(R["MG_2-3-4"][b]["ppl"], 2)) for b in ("2", "3", "4")})
print("gate mass vs uniform vs reverse PPL:", {b: (round(R["gate_mass"][b]["ppl"], 2), round(U[b]["ppl"], 2), round(R["reverse"][b]["ppl"], 2)) for b in MB})
print("mean accuracy gate mass vs uniform:", {b: (round(R["gate_mass"][b]["avg"], 2), round(U[b]["avg"], 2)) for b in MB}, "uniform 4-bit", round(U["4"]["avg"], 2))
