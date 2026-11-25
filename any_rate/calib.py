"""Calibration data and Hessian collection."""
import torch

C4_FILE = "en/c4-train.00000-of-01024.json.gz"


def c4_matgptq(tok, n=128, seqlen=2048):
    """MatGPTQ's C4 calibration: the first documents of C4 training shard 0 in file order, tokenised without special tokens and joined
    with "\\n\\n"; each sample is the first `seqlen` tokens of a joined chunk. Returns n tensors of shape (1, seqlen)."""
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", data_files={"train": C4_FILE}, split="train")
    sep = tok("\n\n", return_tensors="pt", add_special_tokens=False).input_ids; out = []; cur = None
    for rec in ds:
        t_ = tok(rec["text"], return_tensors="pt", add_special_tokens=False).input_ids; cur = t_ if cur is None else torch.cat([cur, t_], 1)
        if cur.numel() >= seqlen: out.append(cur[:, :seqlen]); cur = None
        else: cur = torch.cat([cur, sep], 1)
        if len(out) >= n: break
    assert len(out) == n, len(out)
    return out


class HessianHooks:
    """Forward pre-hooks that accumulate H = sum x x^T (float32) of the inputs of the given modules; key -> module.
    mask(key) may return a boolean mask of the valid positions of the current call (padding is excluded)."""
    def __init__(self, modules, dev, mask=None):
        self.H, self.count, self.dev, self.mask, self.hooks = {}, {}, dev, mask, []
        for key, mod in modules.items(): self.hooks.append(mod.register_forward_pre_hook(self._make(key)))

    def _make(self, key):
        def pre(mod, inp):
            x = inp[0].detach(); m = self.mask(key) if self.mask else None
            x = (x[m.bool()] if m is not None else x.reshape(-1, x.shape[-1])).float()
            if key not in self.H: self.H[key] = torch.zeros(x.shape[1], x.shape[1], device=self.dev); self.count[key] = 0
            self.H[key].addmm_(x.T, x); self.count[key] += x.shape[0]
        return pre

    def remove(self):
        for h in self.hooks: h.remove()
        self.hooks = []
