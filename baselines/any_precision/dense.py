"""Dense weights of one mode of an Any-Precision LLM checkpoint (encode.py): W_b[row] = LUT_b[row][code >> (parent - b)]."""
import glob, os, re
import torch


def weights(ckpt, bits, device="cuda"):
    """yields (module name, fp32 weight) of every quantized layer in the b-bit mode (seed <= b <= parent)"""
    run = ckpt if os.path.isdir(os.path.join(ckpt, "weights")) else glob.glob(os.path.join(ckpt, "quantized", "*"))[0]
    parent = int(re.search(r"-w(\d+)_orig", os.path.basename(run)).group(1))          # the run folder is named (<model>)-w<parent>_orig<seed>-...
    b = int(bits); n = len(glob.glob(os.path.join(run, "weights", "l*.pt")))
    for li in range(n):
        q = torch.load(os.path.join(run, "weights", f"l{li}.pt"), weights_only=False); lut = torch.load(os.path.join(run, f"lut_{b}", f"l{li}.pt"), weights_only=False)
        for name in q:
            codes = torch.from_numpy(q[name][:, 0, :]).to(device).long() >> (parent - b)
            yield f"model.layers.{li}.{name}", torch.gather(torch.from_numpy(lut[name][:, 0, :]).to(device).float(), 1, codes)
