"""Language models (the four of the main results and Qwen3-30B-A3B): WikiText-2 perplexity and the eight benchmarks of every mode."""
import json, os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from baselines.common import original
from . import weights
from .tasks import TASKS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOC = os.path.join(REPO, "results", "I.Mixture-of-Experts_Model", "allocation")
COLS = [("PPL", "ppl", "wikitext2"), ("ARC-C", "csr", "arc_challenge"), ("ARC-E", "csr", "arc_easy"), ("HellaSwag", "csr", "hellaswag"),
        ("PIQA", "csr", "piqa_baber"), ("WinoGrande", "csr", "winogrande"), ("GSM8K", "gsm8k", "flexible"), ("TriviaQA", "triviaqa", "em"), ("NQ-open", "nqopen", "em")]


def expert_modes(kind, b):
    """Qwen3-30B-A3B: the mode of every expert at mean bit width b, allocated by gate mass (greedy) or in reverse (results/H)"""
    alloc = json.load(open(os.path.join(ALLOC, f"b{b:.1f}_{kind}.json")))
    return lambda name: alloc[name.split(".")[2]][name.split(".")[5]]          # model.layers.<l>.mlp.experts.<e>.<proj>


def table(root, dirs):
    lines = ["| method | bits | " + " | ".join(c for c, _, _ in COLS) + " | Avg8 |", "|" + "---|" * (len(COLS) + 3)]
    for d in dirs:
        for key in sorted(os.listdir(os.path.join(root, d)) if os.path.isdir(os.path.join(root, d)) else [], key=lambda k: float(k[:-3]) if k.endswith("bit") else 99):
            v = []
            for _, task, k in COLS:
                f = os.path.join(root, d, key, task, f"{key}_summary.json"); v.append(json.load(open(f))[k] if os.path.exists(f) else None)
            s = [f"{v[0]:.2f}" if v[0] is not None else "–"] + [f"{100 * x:.2f}" if x is not None else "–" for x in v[1:]]
            avg = f"{100 * sum(v[1:]) / 8:.2f}" if None not in v[1:] else "–"
            lines.append(f"| {d} | {key} | " + " | ".join(s) + f" | {avg} |")
    return "\n".join(lines)


def run(model_key, method, ckpt, base, modes, tasks, root, dirs, limit, log):
    moe = model_key.startswith("qwen3"); gemma = model_key == "gemma-3-4b-it"; d = dirs[method]
    key = lambda b: ("bf16" if moe else "fp16") if method == "fp16" else f"{b:g}bit"
    todo = [(b, t) for b in modes for t in tasks if not os.path.exists(os.path.join(root, d, key(b), t, f"{key(b)}_summary.json"))]
    if todo:
        src = original(model_key, base); dt = torch.bfloat16 if gemma or moe else torch.float16
        tok = AutoTokenizer.from_pretrained(src); tok.padding_side = "left"
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dt, **({"attn_implementation": "eager"} if gemma else {})).cuda().eval()
        for b in dict.fromkeys(b for b, _ in todo):
            if method != "fp16":
                bits = expert_modes("greedy" if method == "ar_gate_mass" else "reverse", b) if method in ("ar_gate_mass", "ar_reverse_gate_mass") else b
                log(f"{d} {key(b)}: {weights.apply(model, weights.of(method, ckpt, bits))} layers set")
            for t in [t for bb, t in todo if bb == b]:
                out = os.path.join(root, d, key(b), t); os.makedirs(out, exist_ok=True); t0 = time.time()
                rows, summ = TASKS[t](model, tok, gemma, limit=limit, **({"batch": 16} if moe and t != "ppl" else {})); summ["seconds"] = time.time() - t0
                with open(os.path.join(out, f"{key(b)}_samples.jsonl"), "w") as f:
                    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
                json.dump(summ, open(os.path.join(out, f"{key(b)}_summary.json"), "w"), indent=2); log(d, key(b), t, summ)
    t = table(root, dirs.values()); open(os.path.join(root, "results_table.md"), "w").write(t + "\n"); print(t)
