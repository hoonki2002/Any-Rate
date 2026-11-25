"""Gemma-3-4b-it as a vision-language model: the language model and the SigLIP vision tower in the same mode describe each image in
one sentence (chat template, greedy, at most 120 new tokens). Results: <root>/responses.json."""
import json, os
import torch


@torch.no_grad()
def run(method, ckpt, base, modes, root, images, log):
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    from . import weights
    if not images: raise SystemExit("--images: the image files to describe")
    f = os.path.join(root, "responses.json"); res = json.load(open(f)) if os.path.exists(f) else {}
    key = lambda b: "16" if method == "fp16" else f"{b:g}"; prompt = "Describe this image in one sentence."
    todo = [b for b in ([16] if method == "fp16" else modes) if any(key(b) not in res.get(os.path.basename(i), {}) for i in images)]
    if not todo: return
    proc = AutoProcessor.from_pretrained(base)
    model = Gemma3ForConditionalGeneration.from_pretrained(base, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    for b in todo:
        if method != "fp16": log(f"{b:g} bits: {weights.apply(model, weights.of(method, ckpt, b))} layers set")
        for i in images:
            msgs = [{"role": "user", "content": [{"type": "image", "path": i}, {"type": "text", "text": prompt}]}]
            inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda", torch.bfloat16)
            out = model.generate(**inp, max_new_tokens=120, do_sample=False)
            res.setdefault(os.path.basename(i), {})[key(b)] = proc.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            log(os.path.basename(i), key(b), res[os.path.basename(i)][key(b)])
        os.makedirs(root, exist_ok=True); json.dump(res, open(f, "w"), ensure_ascii=False, indent=1)
