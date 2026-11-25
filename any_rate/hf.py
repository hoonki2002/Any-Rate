"""Load an Any-Rate checkpoint (local folder or Hugging Face repo) as dense weights of one mode.
  state_dict(repo, bits)     -> {name: tensor} with every encoded layer decoded to `dtype`, ready for the original model class
  load_model(repo, bits)     -> the original transformers model (class from config.architectures) with those weights
  load_pipeline(repo, bits)  -> a diffusers pipeline whose encoded components (subfolders with any_rate_config.json) are decoded"""
import glob, json, os
import torch
from safetensors import safe_open
from .format import decode, MODES, RESIDUAL_BITS


def local_path(repo, revision=None):
    if os.path.isdir(repo): return repo
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=revision)


def config(repo, revision=None):
    return json.load(open(os.path.join(local_path(repo, revision), "any_rate_config.json")))


def state_dict(repo, bits, dtype=torch.float16, decode_device=None, revision=None):
    bits = float(bits)
    if bits not in MODES and bits not in RESIDUAL_BITS: raise ValueError(f"bits must be one of {sorted(MODES) + list(RESIDUAL_BITS)}")
    path = local_path(repo, revision); dev = decode_device or ("cuda" if torch.cuda.is_available() else "cpu")
    files = [safe_open(fn, "pt", device="cpu") for fn in sorted(glob.glob(os.path.join(path, "*.safetensors")))]
    where = {k: f for f in files for k in f.keys()}             # the tensors of one layer may sit in different shards
    out, enc = {}, {}
    for k in where:
        if ".ar." in k:
            name, field = k.split(".ar."); enc.setdefault(name, []).append(field)
        else:
            t = where[k].get_tensor(k); out[k] = t.to(dtype) if t.is_floating_point() else t
    for name, fields in enc.items():
        t = {f: where[f"{name}.ar.{f}"].get_tensor(f"{name}.ar.{f}").to(dev) for f in fields if bits > 4 or not f.startswith("res_")}
        out[f"{name}.weight"] = decode(t, bits).to(dtype).cpu()
    return out


def load_model(repo, bits, dtype=torch.float16, device="cuda", revision=None, **kwargs):
    import transformers
    path = local_path(repo, revision); cfg = transformers.AutoConfig.from_pretrained(path)
    cls = getattr(transformers, cfg.architectures[0])
    sd = state_dict(path, bits, dtype=dtype)
    model = cls.from_pretrained(None, config=cfg, state_dict=sd, torch_dtype=dtype, **kwargs)
    return model.to(device).eval()


def load_pipeline(repo, bits, dtype=torch.float16, device="cuda", revision=None, **kwargs):
    import importlib, diffusers
    path = local_path(repo, revision); index = json.load(open(os.path.join(path, "model_index.json"))); parts = {}
    for name, spec in index.items():
        sub = os.path.join(path, name)
        if not (isinstance(spec, list) and os.path.exists(os.path.join(sub, "any_rate_config.json"))): continue
        cls = getattr(importlib.import_module(spec[0]), spec[1]); sd = state_dict(sub, bits, dtype=dtype)
        if spec[0] == "transformers":
            import transformers
            parts[name] = cls.from_pretrained(None, config=transformers.AutoConfig.from_pretrained(sub), state_dict=sd, torch_dtype=dtype)
        else:
            m = cls.from_config(cls.load_config(sub)).to(dtype); m.load_state_dict(sd, strict=True); parts[name] = m
    return diffusers.DiffusionPipeline.from_pretrained(path, torch_dtype=dtype, **parts, **kwargs).to(device)
