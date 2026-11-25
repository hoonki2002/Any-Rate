"""Candidate search tables of the Encoder Design Choices appendix: UHHHHU (proxy loss and time relative to evaluating all candidates, WikiText-2 PPL
of full enumeration vs the beam B = 64) and the search time of UHHHHHH relative to B = 256. 12 matrices of Qwen2.5-3B (k, o, gate, down of layers
3, 18, 33), 128 random rows each, equal mode weights, CPU. PPL: ppl_full_enumeration/ and the B = 64 encoding of the layout table (../ppl/UHHHHU).
search_runs*.json: per matrix and search, relative proxy loss of each mode and time (s).   usage: python table.py"""
import glob, json, math, os
HERE = os.path.dirname(os.path.abspath(__file__)); gm = lambda v: math.exp(sum(map(math.log, v)) / len(v))
runs = json.load(open(f"{HERE}/search_runs.json")) + json.load(open(f"{HERE}/search_runs_extra.json"))
def pick(lay, kind, B): return {r["data"]: r for r in runs if r["layout"] == lay and r["beam"] == B and r.get("kind", "full" if B == 0 else "beam") == kind}
def rel(X, ref, ms): ks = [k for k in ref if k in X]; return [gm([X[k]["rel_err"][m] / ref[k]["rel_err"][m] for k in ks]) for m in ms] + [gm([X[k]["sec"] / ref[k]["sec"] for k in ks])]
ms = ["2", "2.5", "3", "4"]; ref = pick("UHHHHU", "naive", 0); print(f"UHHHHU, {len(ref)} matrices: proxy loss 2 / 2.5 / 3 / 4 and time, relative to evaluating all candidates")
for name, X in [("all candidates", ref), ("full enumeration, top-K", pick("UHHHHU", "full", 0))] + [(f"prefix beam B={B}", pick("UHHHHU", "beam", B)) for B in (16, 32, 64, 128, 256)]:
    print(f"  {name:24s}" + " ".join(f"{v:.3f}" for v in rel(X, ref, ms)[:-1]) + f"  {rel(X, ref, ms)[-1]:.2f}x")
ppl = lambda d: {os.path.basename(f).split("bit")[0]: json.load(open(f))["wikitext2"] for f in glob.glob(f"{HERE}/{d}/*bit_summary.json")}
for name, d in (("full enumeration, top-K", "ppl_full_enumeration"), ("prefix beam B=64", "../ppl/UHHHHU")):
    p = ppl(d); print(f"  PPL {name:24s}" + " ".join(f"{p[b]:.2f}" for b in ms))
ref = pick("UHHHHHH", "beam", 256); print("UHHHHHH: search time relative to B=256")
for B in (16, 64): print(f"  B={B}: {rel(pick('UHHHHHH', 'beam', B), ref, ['2'])[-1]:.2f}x")
