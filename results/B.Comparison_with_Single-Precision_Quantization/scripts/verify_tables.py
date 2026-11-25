"""Check the single-precision GPTQ results end to end:  (1) every cell of <Model>/results_table.csv against the *_summary.json files,  (2) every benchmark
summary against the mean of its per-question records (*_samples.jsonl),  (3) every multiple-choice record: acc == (pred == gold).  usage: python verify_tables.py"""
import os, csv, json
RF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ("Qwen2.5-3B-base", "Gemma-3-4b-it", "Llama-3.1-8B-Instruct", "Phi-4")
T5 = ("arc_challenge", "arc_easy", "hellaswag", "piqa_baber", "winogrande")
COLS = ["PPL(WikiText-2)", "ARC-C", "ARC-E", "HellaSwag", "PIQA", "WinoGrande", "GSM8K(flexible)", "TriviaQA", "NQ-open"]
def rj(p): return json.load(open(p))
def rl(p): return [json.loads(l) for l in open(p)]
def mean(x): return sum(x) / len(x)
bad, n_cells, n_summ, n_rec = [], 0, 0, 0
for model in MODELS:
    for r in csv.DictReader(open(f"{RF}/{model}/results_table.csv")):
        k = f"{r['bit']}bit"; d = f"{RF}/{model}/per_item_grading/GPTQ/{k}"; tag = (model, r["method"], r["bit"])
        S = {t: rj(f"{d}/{t}/{k}_summary.json") for t in ("ppl", "csr", "gsm8k", "triviaqa", "nqopen")}
        # (2) summaries vs per-question records
        cs = rl(f"{d}/csr/{k}_samples.jsonl"); n_rec += len(cs)
        for t in T5:
            rows = [x for x in cs if x["task"] == t]; v = mean([x["acc" if t == "winogrande" else "acc_norm"] for x in rows]); n_summ += 1
            if abs(v - S["csr"][t]) > 1e-9: bad.append((*tag, "csr/" + t, v, S["csr"][t]))
        bad += [(*tag, "csr record", x["task"], x["doc_id"]) for x in cs if x["acc"] != float(x["pred"] == x["gold"])]   # (3)
        for t, key, f in (("gsm8k", "flexible", "correct_flexible"), ("triviaqa", "em", "passed"), ("nqopen", "em", "passed")):
            rows = rl(f"{d}/{t}/{k}_samples.jsonl"); n_summ += 1; v = mean([bool(x[f]) for x in rows])
            if abs(v - S[t][key]) > 1e-9 or len(rows) != S[t]["n"]: bad.append((*tag, t, v, S[t][key]))
        # (1) table cells vs summaries
        vals = [S["ppl"]["wikitext2"], *[100 * S["csr"][t] for t in T5], 100 * S["gsm8k"]["flexible"], 100 * S["triviaqa"]["em"], 100 * S["nqopen"]["em"]]
        for col, v in zip(COLS + ["Avg8"], vals + [mean(vals[1:])]):
            n_cells += 1
            if abs(round(v, 2) - float(r[col])) > 0.011: bad.append((*tag, col, round(v, 2), r[col]))
print(f"table cells: {n_cells}; summaries vs records: {n_summ}; multiple-choice records: {n_rec}; mismatches: {len(bad)}")
for b in bad[:30]: print("  MISMATCH", b)
