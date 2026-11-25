"""The numbers of the appendix comparison with single-precision quantization: the averages over the four models (figure, panels a and b) for
single-precision GPTQ (this folder), MG (2/3/4) and Any-Rate (../A.Full_Benchmark_Results), the per-model mean accuracy behind the text, and the
storage of serving 2-4 and 2-8 bits on Llama-3.1-8B-Instruct (panel c; counted as in ../H.Storage_and_Decoding_Efficiency/storage).
usage: python numbers.py"""
import csv, math, os
RF = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); A = os.path.join(RF, "..", "A.Full_Benchmark_Results")
MODELS = ("Qwen2.5-3B-base", "Gemma-3-4b-it", "Llama-3.1-8B-Instruct", "Phi-4")
R = {}                                                                     # (method, bit) -> {model: (PPL, Avg8)}
for m in MODELS:
    for r in csv.DictReader(open(os.path.join(A, m, "results_table.csv"))):
        if r["method"] in ("FP16", "MG (2/3/4)", "AR (ours)"): R.setdefault((r["method"], float(r["bit"])), {})[m] = (float(r["PPL(WikiText-2)"]), float(r["Avg8"]))
    for r in csv.DictReader(open(os.path.join(RF, m, "results_table.csv"))):
        R.setdefault(("GPTQ", float(r["bit"])), {})[m] = (float(r["PPL(WikiText-2)"]), float(r["Avg8"]))
fp = R[("FP16", 16.0)]
print("mean over the four models: PPL / FP16 PPL (geometric mean) and accuracy in % of FP16")
for meth in ("GPTQ", "MG (2/3/4)", "AR (ours)"):
    for b in (2, 2.5, 3, 3.5, 4):
        if (meth, b) not in R: continue
        d = R[(meth, b)]; p = math.exp(sum(math.log(d[m][0] / fp[m][0]) for m in MODELS) / 4); a = sum(100 * d[m][1] / fp[m][1] for m in MODELS) / 4
        print(f"  {meth:11s} {b:>3g} bits: PPL ratio {p:9.3f}   accuracy {a:5.1f}%")
print("\nmean accuracy (Avg8) per model")
for m in MODELS:
    print(f"  {m:22s}", "  ".join(f"{b:g}b GPTQ {R[('GPTQ', b)][m][1]:.2f} MG {R[('MG (2/3/4)', b)][m][1]:.2f} AR {R[('AR (ours)', b)][m][1]:.2f}" for b in (2.0, 3.0, 4.0)),
          f"| AR 2.5b {R[('AR (ours)', 2.5)][m][1]:.2f} 3.5b {R[('AR (ours)', 3.5)][m][1]:.2f}")
# storage, Llama-3.1-8B-Instruct, 224 linear layers (shapes and conventions of the storage table)
SH = [(4096, 4096), (1024, 4096), (1024, 4096), (4096, 4096), (14336, 4096), (14336, 4096), (4096, 14336)]
N = 32 * sum(r * c for r, c in SH); COLS = 32 * sum(c for _, c in SH); G = 128
bpw = lambda nbytes: 8 * nbytes / N
gptq = lambda b: bpw(N * b / 8 + N // G * 2 + N // G * 2 + COLS * 8)        # b-bit codes, fp16 scale and zero per group, int64 act-order permutation
mg234 = bpw(N * 4 / 8 + N // G * 2 + N // G * 2 + COLS * 8)                # 4-bit master with the same metadata
print(f"\nstorage (bits per weight): GPTQ, one model per bit width: 2-4 bits {sum(gptq(b) for b in (2, 3, 4)):.2f}, 2-8 bits {sum(gptq(b) for b in range(2, 9)):.2f}; "
      f"MG (2/3/4): {mg234:.2f}; Any-Rate: see ../H.Storage_and_Decoding_Efficiency/storage (ladder 4.21, 2-8 bits 8.30)")
