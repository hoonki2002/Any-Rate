"""Any-Rate (AR) checkpoint format and decoding (pure PyTorch, no custom kernel).

Every encoded linear layer `<name>` (weight of shape rows x cols) is stored as the tensors
  <name>.ar.plane0 ... <name>.ar.plane6   the ladder UHHHHHH, most significant plane first
                                          U: 1 bit per weight, uint8 (rows, cols/8), column j of a byte = bit j
                                          H: one Hamming [8,4] codeword number (4 bits) per 8 weights, uint8 (rows, cols/16), two per byte (low nibble first)
  <name>.ar.table_codes, .ar.table_lo, .ar.table_step
                                          one 128-entry table per row: value = lo + code * step (uint8 codes, fp16 lo/step of shape rows x 1)
  <name>.ar.res_planes, .ar.res_lev       residual for the 5-8-bit modes: 4 uncoded planes uint8 (4, rows, cols/8), bit order most significant first,
                                          and per-row offsets of every residual prefix, fp16 (rows, 2+4+8+16)
Mode with K planes (K = 3..7 -> 2, 2.5, 3, 3.5, 4 bits) reads the first K planes; its table is the mean of the 2^(7-K) children of the stored table.
Mode 4+k bits (k = 1..4 -> 5, 6, 7, 8 bits) = the 4-bit weight + offset of the first k residual planes."""
import torch

LAYOUT = "UHHHHHH"
MODES = {2.0: 3, 2.5: 4, 3.0: 5, 3.5: 6, 4.0: 7}          # bits -> number of ladder planes
RESIDUAL_BITS = (5.0, 6.0, 7.0, 8.0)                       # 4 + k residual planes, k = 1..4
_RES_OFF = {1: (0, 2), 2: (2, 6), 3: (6, 14), 4: (14, 30)}
_HAM8 = (0, 240, 204, 60, 170, 90, 102, 150, 255, 15, 51, 195, 85, 165, 153, 105)   # Hamming [8,4] codewords, bit j = 1 <=> sign -1 at column j


def _indices(t, K, rows, cols, layout=LAYOUT):
    """ladder index of every weight from the first K planes (plane p at bit K-1-p)"""
    dev = t["plane0"].device; sh = torch.arange(8, device=dev); pat = torch.tensor(_HAM8, device=dev)
    I = torch.zeros(rows, cols, dtype=torch.long, device=dev)
    for p in range(K):
        x = t[f"plane{p}"].long()
        if layout[p] == "U":
            bit = ((x[..., None] >> sh) & 1).reshape(rows, cols)
        else:
            cid = torch.stack([x & 15, x >> 4], -1).reshape(rows, cols // 8)
            bit = ((pat[cid][..., None] >> sh) & 1).reshape(rows, cols)
        I |= bit << (K - 1 - p)
    return I


def decode_planes(t, layout, K):
    """dense fp32 weight of the mode that reads the first K planes of any layout (used by the encoder design experiments)"""
    b = len(layout); L = 2 ** b; rows = t["plane0"].shape[0]; cols = t["plane0"].shape[1] * (8 if layout[0] == "U" else 16)
    T = t["table_lo"].float() + t["table_codes"].float() * t["table_step"].float()
    g = 2 ** (b - K); par = torch.arange(L, device=T.device) // g
    B = (par[:, None] == par[None, :]).float() / g
    return (T @ B)[:, ::g].contiguous().gather(1, _indices(t, b, rows, cols, layout) >> (b - K))


def decode_residual(t, w4, k):
    """the 4 + k-bit weight: the 4-bit weight w4 plus the offset of the first k residual bits"""
    rows, cols = w4.shape; P = t["res_planes"][:k].long(); sh = torch.arange(7, -1, -1, device=P.device)
    bt = ((P[..., None] >> sh) & 1).reshape(k, rows, cols)
    pre = sum(bt[p] << (k - 1 - p) for p in range(k)); a, c = _RES_OFF[k]
    return w4 + t["res_lev"][:, a:c].float().gather(1, pre)


def decode(t, bits):
    """dense fp32 weight (rows, cols) of one encoded layer in the given mode; t maps 'plane0'.., 'table_*', 'res_*' to tensors"""
    bits = float(bits)
    if bits not in MODES and bits not in RESIDUAL_BITS: raise ValueError(f"{bits} bits is not a mode of {LAYOUT} (modes {sorted(MODES) + list(RESIDUAL_BITS)})")
    w = decode_planes(t, LAYOUT, MODES[min(bits, 4.0)])
    k = int(bits) - 4 if bits > 4 else 0
    return decode_residual(t, w, k) if k else w
