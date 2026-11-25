"""Encode one linear layer into the tensors of the Any-Rate checkpoint format (see any_rate/format.py).

  ar.plane0 .. ar.plane6     ladder UHHHHHH; U: uint8 (rows, cols/8), column j of a byte = bit j;
                             H: uint8 (rows, cols/16), the 4-bit codeword number of each block of 8 columns, two per byte (low nibble first)
  ar.table_codes/lo/step     the 8-bit table, 128 entries per row
  ar.res_planes, ar.res_lev  four residual planes uint8 (4, rows, cols/8), most significant bit first; 30 fp16 offsets per row
"""
import torch
from .ladder import encode_ladder, hamming8, prep
from .residual import encode_residual

LAYOUT, FIRST, WEIGHTS = "UHHHHHH", 3, (1, 1, 2, 3, 4)
ROW_CHUNK = 20000                      # rows encoded together; the best of the N_ref rounds is chosen per chunk
DAMPS = (0.01, 0.1, 1.0)               # Hessian damping, increased only if the damped Hessian is not positive definite


def _codeword_ids(dev):
    C = hamming8().to(dev)
    pat = ((C < 0).long() * (1 << torch.arange(8, device=dev))).sum(1)
    key2id = torch.full((256,), -1, dtype=torch.long, device=dev); key2id[pat] = torch.arange(16, device=dev)
    return key2id


def pack_ladder(I, layout):
    rows, cols = I.shape; b = len(layout); dev = I.device
    key2id = _codeword_ids(dev); sh = torch.arange(8, device=dev); out = {}
    for p, ch in enumerate(layout):
        blk = ((I >> (b - 1 - p)) & 1).view(rows, cols // 8, 8)
        if ch == "U":
            out[f"plane{p}"] = (blk << sh).sum(-1).to(torch.uint8)
        else:
            cid = key2id[(blk << sh).sum(-1)]
            assert bool((cid >= 0).all()), "a coded plane holds a pattern that is not a codeword"
            out[f"plane{p}"] = (cid.view(rows, cols // 16, 2) << torch.tensor([0, 4], device=dev)).sum(-1).to(torch.uint8)
    return out


def pack_residual(R):
    w = (2 ** torch.arange(7, -1, -1, device=R.device)).to(torch.uint8)
    return torch.stack([(((R >> (3 - p)) & 1).to(torch.uint8).reshape(R.shape[0], -1, 8) * w).sum(-1).to(torch.uint8) for p in range(4)])


def _encode_once(W, H, damp, layout, first, weights, beam, K, target, residual):
    U = prep(H, damp); parts = []
    for r0 in range(0, W.shape[0], ROW_CHUNK):
        Wc = W[r0:r0 + ROW_CHUNK]
        I, (codes, lo, step), rec, _ = encode_ladder(Wc, H, layout, first, weights, damp, beam, K, target)
        t = pack_ladder(I, layout); t.update(table_codes=codes, table_lo=lo, table_step=step)
        if residual:
            R, lev = encode_residual(Wc, U, rec[len(layout)]); t.update(res_planes=pack_residual(R), res_lev=lev); del R
        parts.append({k: v.cpu() for k, v in t.items()}); del I, rec; torch.cuda.empty_cache()
    return {k: torch.cat([p[k] for p in parts], 1 if k == "res_planes" else 0) for k in parts[0]}


@torch.no_grad()
def encode_layer(W, H, layout=LAYOUT, first=FIRST, weights=WEIGHTS, beam=64, K=64, target="per_mode", residual=True):
    """W (rows, cols) float32 on the GPU, H (cols, cols) the input Hessian. Returns {'plane0': .., 'res_lev': ..} (CPU tensors).
    The defaults are the paper's settings; the other values are the variants of the encoder design experiments
    (layout, lowest mode `first` = planes it reads, mode weights, beam / K, target rule; see core/ladder.py)."""
    for damp in DAMPS:
        try:
            return _encode_once(W, H, damp, layout, first, weights, beam, K, target, residual)
        except torch.linalg.LinAlgError:
            print(f"Cholesky failed at damping {damp}; retrying with a larger damping", flush=True)
    raise RuntimeError("Cholesky failed for every damping value")
