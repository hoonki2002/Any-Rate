"""Diffusion pipelines: PixArt-Sigma and CogVideoX-2B (diffusion transformer and T5 text encoder of each). One calibration pass with
forward hooks collects the Hessians of all target linear layers; then every layer is encoded. model_dir is the pipeline folder.
  PixArt-Sigma DiT : 128 prompts of Gustavosta/Stable-Diffusion-Prompts (random.seed(0)), full generation, 20 steps, guidance 4.5
  CogVideoX-2B DiT : 32 prompts (data/cogvideox_calibration_prompts.json), 49 frames, 50 steps, guidance 6; 4096 random tokens per call
  T5 (both)        : the 128 Stable-Diffusion prompts, padded to 300 tokens; padding positions are excluded
                     (the two pipelines share the same T5 weights and tokenizer, so their encodings are the same)
"""
import json, os, random
import torch
from ..calib import HessianHooks

PIXART_DIT = [("a1qkv", "attn1.to_q", ["attn1.to_q", "attn1.to_k", "attn1.to_v"], None), ("a1o", "attn1.to_out.0", ["attn1.to_out.0"], None),
              ("a2q", "attn2.to_q", ["attn2.to_q"], None), ("a2kv", "attn2.to_k", ["attn2.to_k", "attn2.to_v"], "emask"),
              ("a2o", "attn2.to_out.0", ["attn2.to_out.0"], None), ("ffi", "ff.net.0.proj", ["ff.net.0.proj"], None), ("ffo", "ff.net.2", ["ff.net.2"], None)]
T5 = [("qkv", "layer.0.SelfAttention.q", ["layer.0.SelfAttention.q", "layer.0.SelfAttention.k", "layer.0.SelfAttention.v"], "tmask"),
      ("o", "layer.0.SelfAttention.o", ["layer.0.SelfAttention.o"], "tmask"),
      ("wi", "layer.1.DenseReluDense.wi_0", ["layer.1.DenseReluDense.wi_0", "layer.1.DenseReluDense.wi_1"], "tmask"),
      ("wo", "layer.1.DenseReluDense.wo", ["layer.1.DenseReluDense.wo"], "tmask")]
COGVIDEOX_DIT = [("a1qkv", "attn1.to_q", ["attn1.to_q", "attn1.to_k", "attn1.to_v"], None), ("a1o", "attn1.to_out.0", ["attn1.to_out.0"], None),
                 ("ffi", "ff.net.0.proj", ["ff.net.0.proj"], None), ("ffo", "ff.net.2", ["ff.net.2"], None)]


def sd_prompts(n=128):
    from datasets import load_dataset
    ds = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split="train"); random.seed(0)
    return [ds[i]["Prompt"] for i in random.sample(range(len(ds)), n)]


def _encode_blocks(net, path, groups, H, blocks, encode_layer, save, log):
    for b in blocks:
        store = {}
        for gk, hp, names, mk in groups:
            assert torch.isfinite(H[(b, gk)]).all(), f"non-finite Hessian {(b, gk)}"
            for nm in names:
                store[f"{path % b}.{nm}"] = encode_layer(net.get_submodule(f"{path % b}.{nm}").weight.data.float(), H[(b, gk)])
            del H[(b, gk)]; torch.cuda.empty_cache()
        save(b, store); log(f"block {b} encoded")


def _masked_hooks(net, path, groups, blocks, dev, g):
    mods = {(b, gk): net.get_submodule(f"{path % b}.{hp}") for b in blocks for gk, hp, names, mk in groups}; mk_of = {gk: mk for gk, hp, names, mk in groups}
    return HessianHooks(mods, dev, mask=lambda key: g.get(mk_of[key[1]]) if mk_of[key[1]] else None)


@torch.no_grad()
def encode_t5(model_dir, encode_layer, save, layers=None, dev="cuda", nsamples=128, log=print):
    """the T5 text encoder of a pipeline (keys encoder.block.<b>...)"""
    from transformers import T5EncoderModel, T5Tokenizer
    prompts = sd_prompts(nsamples); tok = T5Tokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    net = T5EncoderModel.from_pretrained(os.path.join(model_dir, "text_encoder"), torch_dtype=torch.float16).to(dev).eval()
    blocks = range(len(net.encoder.block)) if layers is None else layers; g = {}; hk = _masked_hooks(net, "encoder.block.%d", T5, blocks, dev, g)
    for i in range(0, nsamples, 8):
        e = tok(prompts[i:i + 8], padding="max_length", max_length=300, truncation=True, return_tensors="pt").to(dev); g["tmask"] = e.attention_mask
        net(e.input_ids, attention_mask=e.attention_mask)
    hk.remove(); _encode_blocks(net, "encoder.block.%d", T5, hk.H, blocks, encode_layer, save, log)


@torch.no_grad()
def encode_pixart_dit(model_dir, encode_layer, save, layers=None, dev="cuda", nsamples=128, log=print):
    """the PixArt-Sigma diffusion transformer (keys transformer_blocks.<b>...)"""
    from diffusers import PixArtSigmaPipeline
    prompts = sd_prompts(nsamples)
    pipe = PixArtSigmaPipeline.from_pretrained(model_dir, torch_dtype=torch.float16).to(dev); pipe.set_progress_bar_config(disable=True); net = pipe.transformer
    blocks = range(len(net.transformer_blocks)) if layers is None else layers; g = {}; hk = _masked_hooks(net, "transformer_blocks.%d", PIXART_DIT, blocks, dev, g)
    h = net.register_forward_pre_hook(lambda m, a, kw: g.__setitem__("emask", kw.get("encoder_attention_mask")), with_kwargs=True)
    for i, p in enumerate(prompts): pipe(p, num_inference_steps=20, guidance_scale=4.5, generator=torch.Generator(dev).manual_seed(i))
    h.remove(); hk.remove(); _encode_blocks(net, "transformer_blocks.%d", PIXART_DIT, hk.H, blocks, encode_layer, save, log)


@torch.no_grad()
def encode_cogvideox_dit(model_dir, encode_layer, save, layers=None, dev="cuda", nsamples=32, steps=50, tokens_per_call=4096, log=print):
    """the CogVideoX-2B diffusion transformer (keys transformer_blocks.<b>...)"""
    from diffusers import CogVideoXPipeline
    prompts = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cogvideox_calibration_prompts.json")))[:nsamples]
    pipe = CogVideoXPipeline.from_pretrained(model_dir, torch_dtype=torch.float16).to(dev); pipe.set_progress_bar_config(disable=True); pipe.vae.enable_tiling()
    net, path = pipe.transformer, "transformer_blocks.%d"; blocks = range(len(net.transformer_blocks)) if layers is None else layers
    H, rng, hooks = {}, torch.Generator(device=dev), []; rng.manual_seed(0)

    def make(key):
        def pre(mod, inp):     # a random subset of the tokens of every call (the Hessian averages over many calls and steps)
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]); x = x[torch.randperm(x.shape[0], device=dev, generator=rng)[:tokens_per_call]].float()
            if key not in H: H[key] = torch.zeros(x.shape[1], x.shape[1], device=dev)
            H[key].addmm_(x.T, x)
        return pre

    for b in range(len(net.transformer_blocks)):          # every block is hooked, so the random token subsets do not depend on `layers`
        for gk, hp, names, mk in COGVIDEOX_DIT: hooks.append(net.get_submodule(f"{path % b}.{hp}").register_forward_pre_hook(make((b, gk))))
    for i, p in enumerate(prompts):
        pipe(prompt=p, num_inference_steps=steps, num_frames=49, guidance_scale=6, generator=torch.Generator(dev).manual_seed(i), output_type="latent")
    for h in hooks: h.remove()
    _encode_blocks(net, path, COGVIDEOX_DIT, H, blocks, encode_layer, save, log)
