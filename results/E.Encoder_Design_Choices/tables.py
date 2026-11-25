"""Layout, beam width and mode weight tables of the Encoder Design Choices appendix (Qwen2.5-3B, C4 calibration).
proxy/<setting>/l<layer>_proxy.json: relative proxy loss of each mode (key = planes read) for the 7 matrices of the layer; ppl/<setting>/: WikiText-2 PPL.
Every setting was encoded separately with the same pipeline; 'default' is UHHHHHH, B = 64, weights 1-1-2-3-4, and 'main' is the proxy of the model
of the main results (same settings, a separate run; its PPL is in ../A.Full_Benchmark_Results).   usage: python tables.py [--default main|default]"""
import csv, glob, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = sys.argv[sys.argv.index("--default") + 1] if "--default" in sys.argv else "main"
gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
def proxy(tag): return {(int(os.path.basename(f)[1:].split("_")[0]), m): v for f in glob.glob(f"{HERE}/proxy/{tag}/l*_proxy.json") for m, v in json.load(open(f)).items()}
def ppl(tag):
    if tag == "main":
        return {float(r["bit"]): float(r["PPL(WikiText-2)"]) for r in csv.DictReader(open(f"{HERE}/../A.Full_Benchmark_Results/Qwen2.5-3B-base/results_table.csv")) if r["method"] == "AR (ours)" and float(r["bit"]) <= 4}
    return {float(os.path.basename(f).split("bit")[0]): json.load(open(f))["wikitext2"] for f in glob.glob(f"{HERE}/ppl/{tag}/*bit_summary.json")}
MODES = {"UHHHHHH": {"3": 2, "4": 2.5, "5": 3, "6": 3.5, "7": 4}, "UHHHHU": {"3": 2, "4": 2.5, "5": 3, "6": 4}, "UUHHHH": {"2": 2, "3": 2.5, "4": 3, "5": 3.5, "6": 4},
         "UUUU": {"2": 2, "3": 3, "4": 4}, "HHHHHHU": {"4": 2, "5": 2.5, "6": 3, "7": 4}}
INV = {b: k for k, b in MODES["UHHHHHH"].items()}; BITS = [2, 2.5, 3, 3.5, 4]
def ratio(tag, ref, modes, keys): d = proxy(tag); keys = [k for k in keys if k in d]; return {b: gm([d[k][pl] / ref[k][INV[b]] for k in keys]) for pl, b in modes.items()}, len(keys)
base = proxy(BASE); out = {"default_row": BASE}
out["layout"] = {lay: {"proxy": ratio(lay if lay != "UHHHHHH" else BASE, base, MODES[lay], list(base))[0], "ppl": ppl(lay if lay != "UHHHHHH" else BASE)} for lay in ("UUUU", "UHHHHU", "HHHHHHU", "UUHHHH", "UHHHHHH")}
b256 = proxy("beam256")
out["beam_width"] = {B: {"proxy": ratio(t, b256, MODES["UHHHHHH"], list(b256))[0], "ppl": ppl(t)} for B, t in ((16, "beam16"), (64, BASE), (256, "beam256"))}
sub = [k for k in base if k[0] in range(3, 36, 4)]
out["mode_weights"] = {}
for w, t in (("1-1-2-3-4", BASE), ("1-1-1-1-1", "w11111"), ("1-1-2-2-2", "w11222"), ("1-2-3-5-7", "w12357"), ("1-1-1-1-4", "w11114"), ("3-2-1-1-1", "w32111"), ("4-1-1-1-1", "w41111"), ("7-5-3-2-1", "w75321")):
    p, n = ratio(t, base, MODES["UHHHHHH"], sub); out["mode_weights"][w] = {"proxy": p, "matrices": n, "ppl": ppl(t)}
json.dump(out, open(f"{HERE}/tables_{BASE}.json", "w"), indent=1)
f = lambda d, fmt: " ".join(fmt.format(d[b]) if b in d else "  --  " for b in BITS)
for sec in ("layout", "beam_width", "mode_weights"):
    print(f"== {sec} (proxy: 2 2.5 3 3.5 4 | PPL)")
    for k, v in out[sec].items(): print(f"  {str(k):10s} {f(v['proxy'], '{:.3f}')} | {f(v['ppl'], '{:6.2f}')}" + (f"  n={v['matrices']}" if "matrices" in v else ""))
