"""Dense weights of one mode of a checkpoint, set into the original model."""
import glob, os
import torch


def ar(ckpt, bits):
    """(module name, fp32 weight) of every encoded layer of an Any-Rate checkpoint folder (one model or one pipeline component).
    bits: a mode, or a function of the module name (mixed precision, e.g. one mode per expert)."""
    from safetensors import safe_open
    from any_rate.format import decode
    files = [safe_open(f, "pt", device="cuda") for f in sorted(glob.glob(os.path.join(ckpt, "*.safetensors")))]
    where = {k: f for f in files for k in f.keys() if ".ar." in k}; layers = {}
    for k in where: layers.setdefault(k.split(".ar.")[0], []).append(k.split(".ar.")[1])
    for name, fields in layers.items():
        b = float(bits(name) if callable(bits) else bits)
        yield name, decode({f: where[f"{name}.ar.{f}"].get_tensor(f"{name}.ar.{f}") for f in fields if b > 4 or not f.startswith("res_")}, b)


def of(method, ckpt, bits):
    if method.startswith("ar"): return ar(ckpt, bits)
    from baselines.any_precision import dense as ap
    from baselines.anybcq import dense as ab
    from baselines.matgptq import dense as mg
    from baselines.gptq import dense as gq
    return {"ap": ap, "ab": ab, "mg234": mg, "mg348": mg, "gptq": gq}[method].weights(ckpt, bits)


def names(method, ckpt):
    """names of the layers a checkpoint folder replaces (Any-Rate, MatGPTQ or GPTQ)"""
    if method.startswith("ar"):
        from safetensors import safe_open
        return sorted({k.split(".ar.")[0] for f in glob.glob(os.path.join(ckpt, "*.safetensors")) for k in safe_open(f, "pt").keys() if ".ar." in k})
    return sorted({os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(ckpt, "**", "data.pt"), recursive=True)})


def apply(model, named):
    """set the weights; a name that is not in the model is tried without 'language_model.' (the Gemma-3 language model on its own)"""
    mods = dict(model.named_modules()); n = 0
    for name, w in named:
        m = mods.get(name, mods.get(name.removeprefix("language_model.")))
        if m is not None: m.weight.data = w.to(m.weight.dtype); n += 1
    return n


def originals(model, ckpt_names):
    """CPU copies of the weights that a mode replaces (to go back to the original model)"""
    mods = dict(model.named_modules())
    return {n: mods[n].weight.data.cpu().clone() for n in ckpt_names if n in mods}
