"""Encode a model with Any-Rate and write a Hugging Face checkpoint folder (format: any_rate/format.py).

  python -m any_rate.encode --model qwen2.5-3b

--model is one of MODELS below. The original checkpoint is downloaded from the Hugging Face Hub (--base: a local copy instead) and the
output is written to <original name>-AR2to8 (--out: another folder). Every encoded layer is first stored in <out>/_work and the run
can be resumed; the final folder keeps every other tensor of the original checkpoint, its shard layout and its config files.
Diffusion pipelines are written component by component (transformer/, text_encoder/); the other components are copied.
"""
import argparse, glob, json, os, shutil, time
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from .core.pack import encode_layer, LAYOUT

C4 = "C4 (MatGPTQ-style): the first C4 training shard, documents in order joined with blank lines, 128 sequences of 2048 tokens"
SD_PROMPTS = "128 prompts of Gustavosta/Stable-Diffusion-Prompts"
# model -> components (subfolder of the checkpoint, encoder, calibration)
MODELS = {
    "qwen2.5-3b": [("", "decoder", C4)],
    "llama-3.1-8b-instruct": [("", "decoder", C4)],
    "phi-4": [("", "decoder", C4)],
    "qwen3-30b-a3b-instruct-2507": [("", "decoder", C4 + "; experts with fewer than 512 tokens get an isotropic prior of 512 tokens")],
    "gemma-3-4b-it": [("", "decoder", C4), ("", "siglip", "128 images of COCO val2017")],
    "whisper-large-v3": [("", "whisper", "128 utterances of LibriSpeech dev-clean (teacher forcing)")],
    "pixart-sigma": [("transformer", "pixart_dit", SD_PROMPTS), ("text_encoder", "t5", SD_PROMPTS + ", padded to 300 tokens")],
    "cogvideox-2b": [("transformer", "cogvideox_dit", "32 text prompts, inputs of every linear layer over 50 denoising steps of 49-frame generation"),
                     ("text_encoder", "t5", SD_PROMPTS + ", padded to 300 tokens")],
}
HF_IDS = {"qwen2.5-3b": "Qwen/Qwen2.5-3B", "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct", "phi-4": "microsoft/phi-4",
          "qwen3-30b-a3b-instruct-2507": "Qwen/Qwen3-30B-A3B-Instruct-2507", "gemma-3-4b-it": "google/gemma-3-4b-it",
          "whisper-large-v3": "openai/whisper-large-v3", "pixart-sigma": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", "cogvideox-2b": "THUDM/CogVideoX-2b"}


def run_encoder(kind, root, sub, work, layers, log, encode_layer=encode_layer, save=None):
    """root: the original checkpoint folder (for diffusion models the pipeline folder), sub: the component encoded ('' for a model).
    encode_layer / save: another layer encoder and its storage (the MatGPTQ baseline reuses this calibration); such a save marks a
    finished layer with <work>/<kind>_<i>.done."""
    done = lambda i: os.path.exists(f"{work}/{kind}_{i}.safetensors") or os.path.exists(f"{work}/{kind}_{i}.done")
    if save is None:
        def save(i, d): save_file({f"{n}.ar.{k}": v.contiguous() for n, t in d.items() for k, v in t.items()}, f"{work}/{kind}_{i}.safetensors")
    from .models import decoder, diffusion, siglip, whisper
    fn = {"decoder": decoder.encode, "siglip": siglip.encode, "whisper": whisper.encode, "t5": diffusion.encode_t5,
          "pixart_dit": diffusion.encode_pixart_dit, "cogvideox_dit": diffusion.encode_cogvideox_dit}[kind]
    cfg = json.load(open(os.path.join(root, sub, "config.json")))
    n = {"decoder": lambda: (cfg.get("text_config") or cfg)["num_hidden_layers"], "siglip": lambda: cfg["vision_config"]["num_hidden_layers"],
         "whisper": lambda: cfg["encoder_layers"]}.get(kind, lambda: cfg["num_layers"])()
    todo = [i for i in (range(n) if layers is None else layers) if not done(i)]
    if todo: fn(root, encode_layer, save, layers=todo, log=log)


def export(base, out, works, model_name, parts):
    """Write the release folder for one checkpoint (component): the AR tensors of the encoded layers, every other original tensor
    in its original shard file, the config files of the original folder, and any_rate_config.json."""
    os.makedirs(out, exist_ok=True); sd = {}
    for w in works:
        for f in sorted(glob.glob(f"{w}/*.safetensors")): sd.update(load_file(f))
    encoded = sorted({k.split(".ar.")[0] for k in sd})
    files = [safe_open(fn, "pt", device="cpu") for fn in sorted(glob.glob(f"{base}/*.safetensors"))]; where = {k: f for f in files for k in f.keys()}
    for n in encoded:
        rows, cols = sd[f"{n}.ar.plane0"].shape[0], sd[f"{n}.ar.plane0"].shape[1] * 8
        assert tuple(where[f"{n}.weight"].get_slice(f"{n}.weight").get_shape()) == (rows, cols), n
    enc_w = {f"{n}.weight" for n in encoded}
    for k in where:
        if k not in enc_w: sd[k] = where[k].get_tensor(k)
    idx = (glob.glob(f"{base}/*.safetensors.index.json") or [""])[0]
    orig = json.load(open(idx))["weight_map"] if idx else {k: os.path.basename(f) for f in glob.glob(f"{base}/*.safetensors") for k in safe_open(f, "pt").keys()}
    shard = {k: orig[k.split(".ar.")[0] + ".weight" if ".ar." in k else k] for k in sd}
    for fn in sorted(set(shard.values())): save_file({k: sd[k] for k in sd if shard[k] == fn}, f"{out}/{fn}", metadata={"format": "pt"})
    if idx:
        json.dump({"metadata": {"total_size": sum(t.numel() * t.element_size() for t in sd.values())}, "weight_map": dict(sorted(shard.items()))}, open(f"{out}/{os.path.basename(idx)}", "w"), indent=2)
    for p in glob.glob(f"{base}/*"):
        b = os.path.basename(p)
        if os.path.isfile(p) and not b.endswith((".safetensors", ".bin", ".pth")) and not (".index" in b and b.endswith(".json")) and b != "README.md": shutil.copy(p, f"{out}/{b}")
    json.dump({"format": "any-rate", "format_version": 1, "base_model": model_name, "layout": LAYOUT, "bits": [2, 2.5, 3, 3.5, 4, 6, 8],
               "encoder": {"mode_weights": "1-1-2-3-4", "prefix_beam": 64, "table": "shared 8-bit, 128 entries per row"}, "parts": parts,
               "encoded_layers": encoded}, open(f"{out}/any_rate_config.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--base", help="local folder of the original checkpoint (default: download it from the Hugging Face Hub)")
    ap.add_argument("--out", help="output folder (default: <original name>-AR2to8)")
    ap.add_argument("--layers", type=int, nargs="*", help="encode only these layer / block indices (all by default); no folder is exported")
    a = ap.parse_args(); log = lambda *x: print(time.strftime("[%H:%M:%S]"), *x, flush=True)
    hf_id = HF_IDS[a.model]; a.out = a.out or hf_id.split("/")[1] + "-AR2to8"
    if a.base is None:
        from huggingface_hub import snapshot_download
        log(f"downloading {hf_id}"); a.base = snapshot_download(hf_id)
    comps = {}
    for sub, kind, calib in MODELS[a.model]:
        comps.setdefault(sub, []).append((kind, calib))
    for sub, kinds in comps.items():
        work = os.path.join(a.out, "_work", sub or "model"); os.makedirs(work, exist_ok=True)
        for kind, calib in kinds:
            log(f"encoding {kind} of {os.path.join(a.base, sub)}"); run_encoder(kind, a.base, sub, work, a.layers, log)
    if a.layers is not None: return
    for sub, kinds in comps.items():
        base = os.path.join(a.base, sub) if sub else a.base; out = os.path.join(a.out, sub) if sub else a.out
        export(base, out, [os.path.join(a.out, "_work", sub or "model")], hf_id, [{"component": k, "calibration": c} for k, c in kinds])
    if any(sub for sub in comps):            # diffusers pipeline: copy the components that are not encoded and the pipeline index
        for p in glob.glob(f"{a.base}/*"):
            b = os.path.basename(p)
            if b in comps or b == "README.md": continue
            (shutil.copytree(p, f"{a.out}/{b}", dirs_exist_ok=True) if os.path.isdir(p) else shutil.copy(p, f"{a.out}/{b}"))
    shutil.rmtree(os.path.join(a.out, "_work")); log("written", a.out)


if __name__ == "__main__":
    main()
