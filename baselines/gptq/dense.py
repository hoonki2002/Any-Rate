"""Dense weights of one bit width of a single-precision GPTQ checkpoint (encode.py): <bits>bit/<module>/data.pt."""
import glob, os
import torch


def weights(ckpt, bits, device="cuda"):
    """yields (module name, fp32 weight) of every quantized layer at `bits` (2, 3 or 4)"""
    for f in sorted(glob.glob(os.path.join(ckpt, f"{int(bits)}bit", "*", "data.pt"))):
        yield os.path.basename(os.path.dirname(f)), torch.load(f, map_location=device)["weight"].float()
