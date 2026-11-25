"""Dense weights of one mode of an AnyBCQ checkpoint (encode.py), as the authors' kernel dequantizes them:
W[k, m] = sum_{i<b} (+-1 from bit k % 32 of qweight[k // 32, i, m]) alpha_b[k // g, i, m] + beta_b[k // g, m]; the layer weight is W^T."""
import glob, os
import torch
from safetensors import safe_open


def weights(ckpt, bits, device="cuda", group=128):
    """yields (module name, fp32 weight) of every quantized layer in the b-bit mode (b = 2, 3, 4)"""
    b = int(bits); files = [safe_open(f, "pt", device=device) for f in sorted(glob.glob(os.path.join(ckpt, "model*.safetensors")))]
    where = {k: f for f in files for k in f.keys()}                # the tensors of one layer may sit in different shards
    sh = torch.arange(32, device=device, dtype=torch.int32)
    for k in [k for k in where if k.endswith(".qweight")]:
        name = k[:-len(".qweight")]; q = where[k].get_tensor(k)
        al = where[f"{name}.alpha_{b}"].get_tensor(f"{name}.alpha_{b}").float(); be = where[f"{name}.beta_{b}"].get_tensor(f"{name}.beta_{b}").float()
        K, M = q.shape[0] * 32, q.shape[2]; w = be.repeat_interleave(group, 0).clone()
        for i in range(b):
            w += (((q[:, i, :][:, None, :] >> sh[None, :, None]) & 1).reshape(K, M).float() * 2 - 1) * al[:, i, :].repeat_interleave(group, 0)
        yield name, w.t().contiguous()
