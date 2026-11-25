"""Shared parts of the baseline scripts: the models, the original checkpoints, the authors' code checked out by setup.sh."""
import os, subprocess, sys, time
from any_rate.encode import HF_IDS

HERE = os.path.dirname(os.path.abspath(__file__)); THIRD = os.path.join(HERE, "third_party")
LLMS = ["qwen2.5-3b", "gemma-3-4b-it", "llama-3.1-8b-instruct", "phi-4"]
log = lambda *x: print(time.strftime("[%H:%M:%S]"), *x, flush=True)


def code(name):
    path = os.path.join(THIRD, name)
    if not os.path.isdir(path): sys.exit(f"{path} is missing: run `bash baselines/setup.sh` first")
    return path


def original(model, base=None):
    """local folder of the original checkpoint (downloaded unless `base` is given). Gemma-3-4b-it: its language model, see gemma_text."""
    if base is None:
        from huggingface_hub import snapshot_download
        log(f"downloading {HF_IDS[model]}"); base = snapshot_download(HF_IDS[model])
    return gemma_text(base) if model == "gemma-3-4b-it" else base


def gemma_text(src, dst="gemma-3-4b-it-text"):
    """The baselines quantize the language model of Gemma-3-4b-it as a text-only Gemma3ForCausalLM checkpoint (bf16), as in the paper."""
    if os.path.exists(os.path.join(dst, "config.json")): return os.path.abspath(dst)
    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    log(f"writing the language model of {src} to {dst}")
    full = Gemma3ForConditionalGeneration.from_pretrained(src, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    lm = full.language_model; lm.config.architectures = ["Gemma3ForCausalLM"]
    lm.save_pretrained(dst); AutoTokenizer.from_pretrained(src).save_pretrained(dst)
    return os.path.abspath(dst)


def out_dir(model, tag, out=None):
    return os.path.abspath(out or f"{HF_IDS[model].split('/')[1]}-{tag}")


def run(cmd, cwd, env=None, logfile=None):
    """run a command of the authors' code; its output goes to `logfile` (and the terminal if None)"""
    log(" ".join(cmd) + (f"  > {logfile}" if logfile else ""))
    e = dict(os.environ, **(env or {}))
    with (open(logfile, "w") if logfile else open(os.devnull, "w")) as f:
        r = subprocess.run(cmd, cwd=cwd, env=e, stdout=f if logfile else None, stderr=subprocess.STDOUT if logfile else None)
    if r.returncode: sys.exit(f"failed with exit code {r.returncode}" + (f", see {logfile}" if logfile else ""))
