"""The SigLIP vision tower of Gemma-3, encoded layer by layer like the decoder layers (models/decoder.py).
Calibration: the first 128 images of COCO val2017 (lmms-lab/COCO-Caption2017), stored as JPEG (quality 92) and read back."""
import io
import torch
from ..calib import HessianHooks

GROUPS = [("attn", "self_attn", ["q_proj", "k_proj", "v_proj"]), ("o", "self_attn", ["out_proj"]), ("mlp", "mlp", ["fc1"]), ("down", "mlp", ["fc2"])]
PREFIX = "vision_tower.vision_model.encoder.layers.{li}."


def coco_images(n=128):
    from datasets import load_dataset
    from PIL import Image
    out = []
    for ex in load_dataset("lmms-lab/COCO-Caption2017", split="val", streaming=True):
        buf = io.BytesIO(); ex["image"].convert("RGB").save(buf, format="JPEG", quality=92); buf.seek(0)
        out.append(Image.open(buf).convert("RGB"))
        if len(out) == n: return out


@torch.no_grad()
def encode(model_dir, encode_layer, save, layers=None, dev="cuda", nsamples=128, log=print):
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    dt = torch.bfloat16
    full = Gemma3ForConditionalGeneration.from_pretrained(model_dir, torch_dtype=dt, attn_implementation="eager").eval()
    vis = full.vision_tower.vision_model; blocks = vis.encoder.layers; todo = set(range(len(blocks)) if layers is None else layers)
    proc = AutoProcessor.from_pretrained(model_dir); imgs = coco_images(nsamples)
    pv = torch.cat([proc.image_processor(images=imgs[i:i + 8], return_tensors="pt").pixel_values for i in range(0, nsamples, 8)])
    emb = vis.embeddings.to(dev); nt = (pv.shape[-1] // vis.config.patch_size) ** 2
    inps = torch.zeros((nsamples, nt, vis.config.hidden_size), dtype=dt, device=dev)
    for j in range(nsamples): inps[j] = emb(pv[j:j + 1].to(dev, dt))[0]
    del emb; outs = torch.zeros_like(inps)
    for li in range(len(blocks)):
        layer = blocks[li].to(dev)
        hk = HessianHooks({key: layer.get_submodule(parent).get_submodule(names[0]) for key, parent, names in GROUPS}, dev) if li in todo else None
        for j in range(nsamples):
            o = layer(inps[j:j + 1], attention_mask=None); outs[j] = (o[0] if isinstance(o, (tuple, list)) else o)[0]
        if hk is not None:
            hk.remove(); store = {}
            for key, parent, names in GROUPS:
                assert torch.isfinite(hk.H[key]).all(), f"non-finite Hessian {key} in layer {li}"
                for name in names:
                    store[f"{parent}.{name}"] = encode_layer(layer.get_submodule(parent).get_submodule(name).weight.data.float(), hk.H[key])
                del hk.H[key]; torch.cuda.empty_cache()
            save(li, {PREFIX.format(li=li) + k: v for k, v in store.items()}); log(f"vision layer {li} encoded")
        blocks[li] = layer.cpu(); del layer; torch.cuda.empty_cache(); inps, outs = outs, inps
        if li >= max(todo): break
