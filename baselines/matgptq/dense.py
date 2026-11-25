"""Dense weights of one mode of a MatGPTQ checkpoint (encode.py). The b-bit slice of the master code q (M bits) is (q >> (M-b)) << (M-b),
with the zero point shifted by (2^(M-b) - 1) / 2; columns are stored in act-order and are permuted back."""
import glob, json, os
import torch


def weights(ckpt, bits, device="cuda"):
    """yields (module name, fp32 weight) of every quantized layer in the b-bit mode (a slice of the recipe, or any b <= master)"""
    cfg = [f for f in (os.path.join(ckpt, "baseline_config.json"), os.path.join(ckpt, "..", "baseline_config.json")) if os.path.exists(f)][0]   # a pipeline component: one level up
    master = json.load(open(cfg))["master_bitwidth"]
    sh = master - int(bits); off = (2 ** sh - 1) / 2
    for f in sorted(glob.glob(os.path.join(ckpt, "**", "data.pt"), recursive=True)):
        d = torch.load(f, map_location=device); q = d["qweight"].long(); rows, cols = q.shape
        qs = ((q >> sh) << sh).float().view(rows, -1, 128)
        w = (d["scale"].float()[:, :, None] * (qs - (d["zero"].float()[:, :, None] - off))).view(rows, cols)[:, d["perm"].long().argsort()]
        yield os.path.basename(os.path.dirname(f)), w
