"""The tasks of the paper. Each returns (per-question rows, summary). Greedy decoding, plain few-shot prompts (no chat template).
  ppl       WikiText-2 test joined with blank lines, non-overlapping 2048-token windows (Gemma-3: <bos> first in every window)
  csr       lm-eval-harness 0.4.5, 0-shot: ARC-C, ARC-E, HellaSwag, PIQA (baber/piqa), WinoGrande; acc_norm (WinoGrande: acc)
  gsm8k     5-shot (the first 5 training questions), flexible-extract exact match, the first-answer rule below
  triviaqa  5-shot, rc.nocontext, the first 1000 validation questions, exact match against the aliases
  nqopen    5-shot, the first 1000 validation questions, exact match"""
import copy, math, os, re, string
from collections import Counter
import torch
import torch.nn.functional as F
from datasets import load_dataset


@torch.no_grad()
def ppl(model, tok, gemma, limit=None):
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]; n = len(ids) // 2048; X = ids[:n * 2048].view(n, 2048).clone(); loss = 0.0
    if gemma: X[:, 0] = tok.bos_token_id
    for i in range(0, n, 1 if gemma else 2):                # Gemma-3: one window per step (262k-token vocabulary)
        x = X[i:i + (1 if gemma else 2)].cuda(); lg = model(x).logits[:, :-1].float()
        loss += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1), reduction="sum").item()
    return [], {"wikitext2": math.exp(loss / (n * 2047))}


def csr(model, tok, gemma, limit=None, batch=32):
    import lm_eval
    names = ["arc_challenge", "arc_easy", "hellaswag", "piqa_baber", "winogrande"]
    lm = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=tok, batch_size=batch, add_bos_token=gemma)
    tm = lm_eval.tasks.TaskManager(include_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lm_eval_tasks"))
    res = lm_eval.simple_evaluate(model=lm, task_manager=tm, tasks=names, num_fewshot=0, batch_size=batch, limit=limit, log_samples=True)
    rows, summ = [], {}
    for t in names:
        summ[t] = res["results"][t]["acc,none" if t == "winogrande" else "acc_norm,none"]
        for s in res["samples"][t]:
            ll = [x[0][0] if isinstance(x[0], (list, tuple)) else x[0] for x in s["resps"]]; pred = max(range(len(ll)), key=lambda i: ll[i])
            gold = s.get("target") if t != "winogrande" else (pred if s["acc"] == 1.0 else 1 - pred)     # WinoGrande: index of the right option
            rows.append({"task": t, "doc_id": s["doc_id"], "gold": gold, "pred": pred, "acc": s.get("acc"), "acc_norm": s.get("acc_norm")})
    summ["avg"] = sum(summ[t] for t in names) / len(names); summ["metric"] = "acc_norm (winogrande: acc)"
    return rows, summ


def _generate(model, tok, prompts, stop, max_new, prefix=None):
    """greedy continuations of `prompts`; prefix = (ids, KV cache) of a few-shot prefix shared by the batch"""
    if prefix is None:
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda"); ids, mask, cache = enc.input_ids, enc.attention_mask, None
    else:
        pre_ids, pre_cache = prefix; suf = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        cache = copy.deepcopy(pre_cache); cache.batch_repeat_interleave(len(prompts)); P = pre_ids.shape[1]
        ids = torch.cat([pre_ids.expand(len(prompts), -1), suf.input_ids], 1)
        mask = torch.cat([torch.ones(len(prompts), P, dtype=suf.attention_mask.dtype, device="cuda"), suf.attention_mask], 1)
    out = model.generate(input_ids=ids, attention_mask=mask, past_key_values=cache, max_new_tokens=max_new, do_sample=False, use_cache=True,
                         stop_strings=stop, tokenizer=tok, pad_token_id=tok.pad_token_id)
    texts = []
    for j in range(len(prompts)):
        g = out[j, ids.shape[1]:]; texts.append(tok.decode(g[g != tok.pad_token_id], skip_special_tokens=True))
    return texts


def _num(s): return s.replace(",", "").rstrip(".")
def _flexible(t):
    m = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", t); return _num("".join(m[-1]).replace("$", "")) if m else None
def _same(a, b):
    if a is None: return False
    if a == b: return True
    try: return float(a) == float(b)
    except ValueError: return False
# first-answer rule: a generation that keeps going for at least 40 characters after its first "#### <number>" is cut there; it is
# regraded only if this changes the extracted answer
_ANS = re.compile(r"####\s*\$?-?[\d,]+(?:\.\d+)?")
def _cut(t): m = _ANS.search(t); return t[:m.end()] if m and len(t[m.end():].strip()) >= 40 else t
def _norm_ans(s): s = re.sub(r"(?s).*#### ", "", s); s = s.replace(",", "").replace("$", ""); s = re.sub(r"\.$", "", s); return s.strip()
def _ext(t): m = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", t); return "[invalid]" if not m else _norm_ans(m[-1][0] or m[-1][1])


@torch.no_grad()
def gsm8k(model, tok, gemma, limit=None, batch=16, max_new=320):
    ds = load_dataset("openai/gsm8k", "main"); train, test = ds["train"], ds["test"]
    test = list(test.select(range(limit)) if limit else test); stop = ["Question:"]
    fewshot = "".join(f"Question: {train[i]['question']}\nAnswer: {train[i]['answer']}\n\n" for i in range(5))
    prefix = None
    if not gemma:       # the few-shot prefix is encoded once and its KV cache shared by the batch (Gemma-3: full prompts)
        pre_ids = tok(fewshot, return_tensors="pt").input_ids.cuda(); prefix = (pre_ids, model(pre_ids, use_cache=True).past_key_values)
    rows = []
    for st in range(0, len(test), batch):
        ch = test[st:st + batch]; qs = [f"Question: {e['question']}\nAnswer:" for e in ch]
        for j, (e, text) in enumerate(zip(ch, _generate(model, tok, [fewshot + q for q in qs] if gemma else qs, stop, max_new, prefix))):
            for s in stop:
                if s in text: text = text[:text.index(s)]
            gold = _num(e["answer"].split("####")[-1].strip()); pred = _flexible(text); ok = _same(pred, gold); c = _cut(text)
            if _ext(c) != _ext(text): pred = _ext(c); ok = pred == _norm_ans(gold)
            rows.append({"idx": st + j, "gold": gold, "generation": c, "pred_flexible": pred, "correct_flexible": bool(ok)})
        print(f"  gsm8k {len(rows)}/{len(test)}", flush=True)
    return rows, {"flexible": sum(r["correct_flexible"] for r in rows) / len(rows), "n": len(rows)}


def _norm(t): return " ".join(re.sub(r"\b(a|an|the)\b", " ", "".join(c for c in t.lower() if c not in set(string.punctuation))).split())
def _f1(pred, golds):
    best = 0.0; p = _norm(pred).split()
    for g in golds:
        gt = _norm(g).split(); c = sum((Counter(p) & Counter(gt)).values())
        if c: pr, rc = c / len(p), c / len(gt); best = max(best, 2 * pr * rc / (pr + rc))
        elif not p and not gt: best = 1.0
    return best


def _short(model, tok, exs, fewshot, prompt, batch, max_new):
    out = []
    for st in range(0, len(exs), batch):
        ch = exs[st:st + batch]; out += [t.split("\n")[0].strip() for t in _generate(model, tok, [fewshot + prompt(e) for e in ch], ["\n"], max_new)]
        print(f"  {len(out)}/{len(exs)}", flush=True)
    return out


@torch.no_grad()
def triviaqa(model, tok, gemma, limit=None, batch=16, max_new=384):
    d = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext"); exs = list(d["validation"].select(range(limit or 1000)))
    fewshot = "".join(f"Question: {e['question']}\nAnswer: {e['answer']['value']}\n\n" for e in d["train"].select(range(5)))
    preds = _short(model, tok, exs, fewshot, lambda e: f"Question: {e['question']}\nAnswer:", batch, max_new); rows = []
    for e, p in zip(exs, preds):
        al = {_norm(a) for a in e["answer"]["aliases"]} | {_norm(e["answer"]["value"])}
        rows.append({"question_id": e["question_id"], "question": e["question"], "prediction": p, "gold": e["answer"]["value"], "passed": _norm(p) in al})
    return rows, {"em": sum(r["passed"] for r in rows) / len(rows), "n": len(rows)}


@torch.no_grad()
def nqopen(model, tok, gemma, limit=None, batch=16, max_new=384):
    d = load_dataset("google-research-datasets/nq_open"); exs = list(d["validation"].select(range(limit or 1000)))
    fewshot = "".join(f"Question: {e['question']}?\nAnswer: {e['answer'][0]}\n\n" for e in d["train"].select(range(5)))
    preds = _short(model, tok, exs, fewshot, lambda e: f"Question: {e['question']}?\nAnswer:", batch, max_new); rows = []
    for e, p in zip(exs, preds):
        rows.append({"question": e["question"], "prediction": p, "gold": e["answer"], "passed": _norm(p) in {_norm(g) for g in e["answer"]}, "f1": _f1(p, e["answer"])})
    return rows, {"em": sum(r["passed"] for r in rows) / len(rows), "f1": sum(r["f1"] for r in rows) / len(rows), "n": len(rows)}


TASKS = {"ppl": ppl, "csr": csr, "gsm8k": gsm8k, "triviaqa": triviaqa, "nqopen": nqopen}
