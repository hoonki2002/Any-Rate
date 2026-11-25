"""Evaluate one method on one model of the paper at every mode.

  python -m evaluation --model qwen2.5-3b --method ar

  language models    WikiText-2 perplexity and the eight benchmarks, then the table (llm.py, tasks.py)
  qwen3-30b-a3b      the same; --method ar_gate_mass / ar_reverse_gate_mass: one mode per expert (results/I.../allocation)
  whisper-large-v3   WER on LibriSpeech test-clean, test-other and Earnings-22 (speech.py)
  pixart-sigma       the images of the paper; cogvideox-2b: the video (generation.py)
  gemma-3-4b-it      with --tasks vlm: one-sentence descriptions of the two photos of the paper (evaluation/images; --images: others),
                     with the vision tower in the same mode (vlm.py)

--method ar reads <original name>-AR2to8 (the output of any_rate.encode) or, if there is no such folder, the checkpoint on Hugging Face;
ap / ab / mg234 / mg348 / gptq read the output of baselines/<method>/encode.py (gptq: single-precision GPTQ, one model per bit width); fp16 is the original model. Results go to eval_results/<model>/;
finished runs are skipped."""
import argparse, glob, os, time
from any_rate.encode import HF_IDS
from .tasks import TASKS

LLM = {"fp16": ("FP16", None, [16]), "ar": ("AR_ours", "AR2to8", [2, 2.5, 3, 3.5, 4, 6, 8]), "ap": ("AP", "AP", [2, 3, 4, 6, 8]),
       "ab": ("AB", "AB2to4", [2, 3, 4]), "mg234": ("MG_2-3-4", "MG234", [2, 3, 4]), "mg348": ("MG_3-4-8", "MG348", [2, 3, 4, 6, 8]),
       "gptq": ("GPTQ", "GPTQ", [2, 3, 4])}
MOE = {"fp16": ("BF16", None, [16]), "ar": ("AR_uniform", "AR2to8", [2, 2.5, 3, 3.5, 4]), "ar_gate_mass": ("AR_gate_mass", "AR2to8", [2.5, 3, 3.5]),
       "ar_reverse_gate_mass": ("AR_reverse_gate_mass", "AR2to8", [2.5, 3, 3.5]), "mg234": ("MG_2-3-4", "MG234", [2, 3, 4])}
OTHER = {"fp16": ("FP16", None, [16]), "ar": ("AR_ours", "AR2to8", [2, 2.5, 3, 3.5, 4]), "mg234": ("MG_2-3-4", "MG234", [2, 3, 4])}
MODELS = {"qwen2.5-3b": ("Qwen2.5-3B-base", LLM), "gemma-3-4b-it": ("Gemma-3-4b-it", LLM), "llama-3.1-8b-instruct": ("Llama-3.1-8B-Instruct", LLM),
          "phi-4": ("Phi-4", LLM), "qwen3-30b-a3b-instruct-2507": ("Qwen3-30B-A3B-Instruct-2507", MOE),
          "whisper-large-v3": ("Whisper-large-v3", {**OTHER, "ar": ("AR_ours", "AR2to8", [2, 2.5, 3, 3.5, 4, 6, 8])}),
          "pixart-sigma": ("PixArt-Sigma", OTHER), "cogvideox-2b": ("CogVideoX-2b", OTHER)}
PPL_ONLY = {("qwen3-30b-a3b-instruct-2507", "ar_reverse_gate_mass"), ("qwen3-30b-a3b-instruct-2507", "mg234")}   # as reported in the paper


def checkpoint(model, method, suffix, ckpt):
    if method == "fp16" or ckpt: return ckpt
    local = f"{HF_IDS[model].split('/')[1]}-{suffix}"
    if os.path.isdir(local): return os.path.abspath(local)
    if method.startswith("ar"):
        from any_rate.hf import local_path
        return local_path(f"HoonkiLee/{local}")
    raise SystemExit(f"{local} not found: encode it first (baselines/README.md) or pass --ckpt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--method", required=True, choices=sorted({m for _, ms in MODELS.values() for m in ms}))
    ap.add_argument("--ckpt", help="checkpoint folder (default: see above)")
    ap.add_argument("--base", help="local folder of the original checkpoint (default: download it)")
    ap.add_argument("--bits", type=float, nargs="*", help="modes (default: those of the paper)")
    ap.add_argument("--tasks", nargs="*", choices=list(TASKS) + ["vlm"], help="language models: tasks (default: all; vlm: Gemma-3 with images)")
    ap.add_argument("--images", nargs="*", help="--tasks vlm: image files (default: the two photos of the paper)")
    ap.add_argument("--out", default="eval_results")
    ap.add_argument("--limit", type=int, help="only the first N questions / utterances, written to <out>_limit<N> (a quick check)")
    a = ap.parse_args(); log = lambda *x: print(time.strftime("[%H:%M:%S]"), *x, flush=True)
    name, methods = MODELS[a.model]
    if a.method not in methods: raise SystemExit(f"--method for {a.model}: {', '.join(methods)}")
    d, suffix, modes = methods[a.method]; modes = a.bits or modes; ckpt = checkpoint(a.model, a.method, suffix, a.ckpt)
    out = a.out + (f"_limit{a.limit}" if a.limit else "")
    def base():
        if a.base: return a.base
        from huggingface_hub import snapshot_download
        return snapshot_download(HF_IDS[a.model])
    if a.tasks == ["vlm"]:
        from . import vlm
        images = a.images or sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "*.jpg")))   # the two photos of the paper
        vlm.run(a.method, ckpt, base(), a.bits or [m for m in modes if m <= 4], os.path.join(out, name + "_vision"), images, log)
    elif methods is LLM or methods is MOE:
        from . import llm
        tasks = a.tasks or (["ppl"] if (a.model, a.method) in PPL_ONLY else list(TASKS)); root = os.path.join(out, name); os.makedirs(root, exist_ok=True)
        llm.run(a.model, a.method, ckpt, a.base, modes, tasks, root, {m: v[0] for m, v in methods.items()}, a.limit, log)
    elif a.model == "whisper-large-v3":
        from . import speech
        speech.run(a.method, ckpt, base(), modes, os.path.join(out, name), d, a.limit, log)
    else:
        from . import generation
        generation.run(a.model, a.method, ckpt, base(), modes, os.path.join(out, name), {"ar": "ar", "mg234": "mg"}.get(a.method, ""), log)


if __name__ == "__main__":
    main()
