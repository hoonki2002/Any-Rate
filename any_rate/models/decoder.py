"""Decoder-only language models, encoded layer by layer: Qwen2.5, Llama-3.1, Phi-4, Gemma-3 (language model) and Qwen3-MoE (routed experts).

The calibration sequences pass through the original model one decoder layer at a time. In each layer, forward pre-hooks collect the
input Hessian of every group of linear layers that share an input, the layer's output is computed with its original weights (so the
quantization error of earlier layers never enters later Hessians), and then every linear layer of the group is encoded.
"""
import torch, torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from ..calib import HessianHooks, c4_matgptq

# groups of linear layers that share an input: (key, parent module, layers)
GROUPS = {
    "dense": [("attn", "self_attn", ["q_proj", "k_proj", "v_proj"]), ("o", "self_attn", ["o_proj"]), ("mlp", "mlp", ["gate_proj", "up_proj"]), ("down", "mlp", ["down_proj"])],
    "phi3": [("attn", "self_attn", ["qkv_proj"]), ("o", "self_attn", ["o_proj"]), ("mlp", "mlp", ["gate_up_proj"]), ("down", "mlp", ["down_proj"])],
}
MOE_PRIOR = 512        # an expert that receives n < MOE_PRIOR calibration tokens gets (MOE_PRIOR / n) (tr H / d) I added to its Hessian


def moe_groups(layer):
    """one Hessian per expert input: gate/up share the routed tokens, down sees act(gate) * up"""
    return [g for e in range(len(layer.mlp.experts)) for g in ((f"e{e}gu", f"mlp.experts.{e}", ["gate_proj", "up_proj"]), (f"e{e}d", f"mlp.experts.{e}", ["down_proj"]))]


def load(model_dir):
    """Returns (language model, tokenizer, kind, dtype, prefix of the decoder layers in the checkpoint)."""
    tok = AutoTokenizer.from_pretrained(model_dir); mt = AutoConfig.from_pretrained(model_dir).model_type
    if mt == "gemma3":            # multimodal checkpoint: the text tower, in bf16 (fp16 activations overflow) with eager attention
        from transformers import Gemma3ForConditionalGeneration
        full = Gemma3ForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager").eval()
        model = full.language_model; del full.vision_tower
        return model, tok, "gemma3", torch.bfloat16, "language_model.model.layers.{li}."
    kind = {"phi3": "phi3", "qwen3_moe": "moe"}.get(mt, "dense"); dt = torch.bfloat16 if kind == "moe" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dt, low_cpu_mem_usage=True).eval()
    return model, tok, kind, dt, "model.layers.{li}."


@torch.no_grad()
def encode(model_dir, encode_layer, save, layers=None, dev="cuda", nsamples=128, seqlen=2048, log=print, moe_prior=MOE_PRIOR, on_layer=None):
    """encode_layer(W, H) -> tensors of one linear layer; save(li, {name: tensors}) stores the encoded decoder layer li.
    `layers`: indices to encode (default all); the others are only pushed through. moe_prior: 0 keeps the raw expert Hessians;
    on_layer(li, layer) -> hook handles: extra hooks during the calibration pass of an encoded layer (e.g. router statistics)."""
    model, tok, kind, dt, prefix = load(model_dir); model.config.use_cache = False
    calib = c4_matgptq(tok, nsamples, seqlen)
    if kind == "gemma3": calib = [torch.cat([torch.tensor([[tok.bos_token_id]]), c[:, 1:]], 1) for c in calib]   # every sequence starts with <bos>
    blocks = model.model.layers; todo = set(range(len(blocks)) if layers is None else layers)
    model.model.embed_tokens.to(dev); model.model.rotary_emb.to(dev); blocks[0] = blocks[0].to(dev)
    if kind == "gemma3": model.model.rotary_emb_local.to(dev)
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dt, device=dev); cache = {"i": 0}

    class Catcher(nn.Module):                      # records the inputs of the first decoder layer and stops the forward pass
        def __init__(s, m): super().__init__(); s.module = m
        def forward(s, x, **kw): inps[cache["i"]] = x; cache["i"] += 1; cache["kw"] = kw; raise ValueError

    blocks[0] = Catcher(blocks[0])
    for ids in calib:
        try: model(ids.to(dev))
        except ValueError: pass
    blocks[0] = blocks[0].module.cpu(); model.model.embed_tokens.cpu()
    kw = {k: v for k, v in cache["kw"].items() if k in ("attention_mask", "position_ids", "cache_position", "position_embeddings", "position_embeddings_global", "position_embeddings_local")}
    outs = torch.zeros_like(inps)
    for li in range(len(blocks)):
        layer = blocks[li].to(dev); groups = moe_groups(layer) if kind == "moe" else GROUPS["phi3" if kind == "phi3" else "dense"]
        hk = HessianHooks({key: layer.get_submodule(parent).get_submodule(names[0]) for key, parent, names in groups}, dev) if li in todo else None
        extra = on_layer(li, layer) if on_layer is not None and li in todo else []
        for j in range(nsamples):
            o = layer(inps[j:j + 1], **kw); outs[j] = (o[0] if isinstance(o, (tuple, list)) else o)[0]
        for h in extra: h.remove()
        if hk is not None:
            hk.remove(); H = hk.H; store = {}
            for key, parent, names in groups:
                d = layer.get_submodule(parent).get_submodule(names[0]).weight.shape[1]
                if key not in H or not H[key].any(): H[key] = torch.eye(d, device=dev); log(f"   no calibration token reached {parent}")
                elif kind == "moe" and hk.count[key] < moe_prior:
                    n = hk.count[key]; H[key] += (moe_prior * torch.trace(H[key]) / (n * d)) * torch.eye(d, device=dev)
                assert torch.isfinite(H[key]).all(), f"non-finite Hessian {key} in layer {li}"
                for name in names:
                    W = layer.get_submodule(parent).get_submodule(name).weight.data.float()
                    store[f"{parent}.{name}"] = encode_layer(W, H[key]); del W
                del H[key]; torch.cuda.empty_cache()
            save(li, {prefix.format(li=li) + k: v for k, v in store.items()})
            log(f"layer {li} encoded")
        blocks[li] = layer.cpu(); del layer; torch.cuda.empty_cache(); inps, outs = outs, inps
        if li >= max(todo): break
