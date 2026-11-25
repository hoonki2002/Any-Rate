"""PixArt-Sigma images and CogVideoX-2B videos of the paper at every mode. The diffusion transformer runs in fp16; the T5 text
encoder in bf16 (fp16 overflows once its weights are quantized), its embeddings cast to fp16. Empty negative prompt.
  PixArt-Sigma : 1024 x 1024, 20 steps, guidance 4.5; the three prompts below with both parts in the mode, and the market prompt also
                 with only the transformer (dit) or only the text encoder (t5) in the mode. <root>/images/<prompt>/<method>_<bits>bit_<parts>_<prompt>.png
  CogVideoX-2B : 49 frames, 50 steps, guidance 6, seed 42, 8 fps. <root>/videos/robot_<method>_<bits>bit_dit+t5.mp4"""
import os, time
import torch
from transformers import T5EncoderModel

PIXART_PROMPTS = {"market": ("A busy street market with many people and colorful stalls", 102),
                  "chihuahua": ("A happy, dark-colored chihuahua with white markings and a visible tongue is lying on a dark surface with a leash attached "
                                "to its collar, looking directly at the camera with a joyful expression.", 102),
                  "sunflowers": ("A bouquet of sunflowers in a glass vase on a wooden table by a sunny window", 1007)}
ROBOT = "A robotic arm picks up a tiny chip and places it onto a circuit board in a clean factory, camera slowly panning"


def _pipeline(cls, base):
    te = T5EncoderModel.from_pretrained(os.path.join(base, "text_encoder"), torch_dtype=torch.bfloat16)
    pipe = cls.from_pretrained(base, text_encoder=te, torch_dtype=torch.float16).to("cuda"); pipe.text_encoder.to(torch.bfloat16)
    pipe.set_progress_bar_config(disable=True); return pipe, te


@torch.no_grad()
def run(model_key, method, ckpt, base, modes, root, tag, log):
    from . import weights
    name = lambda b: "fp16" if method == "fp16" else f"{tag}_{b:g}bit"
    if model_key == "pixart-sigma":
        from diffusers import PixArtSigmaPipeline
        jobs = [(b, p, parts) for b in modes for p in PIXART_PROMPTS for parts in (["dit+t5"] + (["dit", "t5"] if p == "market" else []))]
        path = lambda b, p, parts: os.path.join(root, "images", p, f"{name(b)}_{p}.png" if method == "fp16" else f"{name(b)}_{parts}_{p}.png")
        if method == "fp16": jobs = [(16, p, "dit+t5") for p in PIXART_PROMPTS]
        jobs = [j for j in jobs if not os.path.exists(path(*j))]
        if not jobs: return
        pipe, te = _pipeline(PixArtSigmaPipeline, base); nets = {"dit": pipe.transformer, "t5": te}
        orig = {p: weights.originals(nets[p], weights.names(method, os.path.join(ckpt, c))) for p, c in (("dit", "transformer"), ("t5", "text_encoder"))} if method != "fp16" else {}
        for b in dict.fromkeys(j[0] for j in jobs):
            for parts in dict.fromkeys(j[2] for j in jobs if j[0] == b):
                if method != "fp16":
                    for p, c in (("dit", "transformer"), ("t5", "text_encoder")):
                        weights.apply(nets[p], weights.of(method, os.path.join(ckpt, c), b) if p in parts.split("+") else ((k, v.cuda()) for k, v in orig[p].items()))
                    log(f"{name(b)} {parts}")
                for _, p, _ in [j for j in jobs if j[0] == b and j[2] == parts]:
                    prompt, seed = PIXART_PROMPTS[p]
                    pe, pm, ne, nm = pipe.encode_prompt(prompt, do_classifier_free_guidance=True, negative_prompt="", device="cuda", clean_caption=False)
                    pipe.text_encoder = None        # with no text encoder, the pipeline keeps the given embeddings in the transformer dtype
                    img = pipe(prompt=None, negative_prompt=None, prompt_embeds=pe.to(torch.float16), prompt_attention_mask=pm, negative_prompt_embeds=ne.to(torch.float16),
                               negative_prompt_attention_mask=nm, num_inference_steps=20, guidance_scale=4.5, height=1024, width=1024,
                               generator=torch.Generator("cuda").manual_seed(seed)).images[0]
                    pipe.text_encoder = te; f = path(b, p, parts); os.makedirs(os.path.dirname(f), exist_ok=True); img.save(f); log("written", f)
    else:
        from diffusers import CogVideoXPipeline
        from diffusers.utils import export_to_video
        path = lambda b: os.path.join(root, "videos", "robot_fp16.mp4" if method == "fp16" else f"robot_{name(b)}_dit+t5.mp4")
        todo = [b for b in ([16] if method == "fp16" else modes) if not os.path.exists(path(b))]
        if not todo: return
        pipe, te = _pipeline(CogVideoXPipeline, base); pipe.vae.enable_tiling()
        for b in todo:
            if method != "fp16":
                n = weights.apply(pipe.transformer, weights.of(method, os.path.join(ckpt, "transformer"), b)) + weights.apply(te, weights.of(method, os.path.join(ckpt, "text_encoder"), b))
                log(f"{name(b)}: {n} layers set")
            t0 = time.time()
            pe, ne = pipe.encode_prompt(ROBOT, negative_prompt="", do_classifier_free_guidance=True, num_videos_per_prompt=1, device="cuda", dtype=torch.bfloat16)
            pipe.text_encoder = None
            frames = pipe(prompt=None, negative_prompt=None, prompt_embeds=pe.to(torch.float16), negative_prompt_embeds=ne.to(torch.float16), num_inference_steps=50,
                          num_frames=49, guidance_scale=6, generator=torch.Generator("cuda").manual_seed(42)).frames[0]
            pipe.text_encoder = te; os.makedirs(os.path.dirname(path(b)), exist_ok=True); export_to_video(frames, path(b), fps=8)
            log("written", path(b), f"({time.time() - t0:.0f}s)")
