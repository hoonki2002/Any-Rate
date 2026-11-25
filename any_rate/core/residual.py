"""Residual planes for the 5-8-bit modes (Appendix D.5 of the paper).

On top of the 4-bit reconstruction q4 of the ladder, every weight stores a 4-bit residual index rho (four uncoded planes, the most
significant first). Mode 4+k adds the offset delta_k[rho >> (4-k)] from a per-row table of 2^k offsets (30 fp16 offsets per row).
The indices are chosen column by column with one target per mode and GPTQ-style error feedback; the offsets start at the centres of
2^k equal intervals of [-m, m] (m = 95th percentile of |w - q4| in the row), are refit once as the mean residual of their weights, and
the indices are chosen again.
"""
import torch

KS = (1, 2, 3, 4)
QUANTILE = 0.95


def mid_levels(m):
    return {k: (torch.arange(2 ** k, device=m.device)[None].float() + 0.5 - 2 ** (k - 1)) * (2 * m / 2 ** k) for k in KS}


def choose(W, U, q4, lev):
    ts = {k: W.clone() for k in KS}; R = torch.empty(W.shape, dtype=torch.long, device=W.device); n = W.shape[1]; cand = torch.arange(16, device=W.device)
    for j in range(n):
        cost = 0
        for k in KS:
            cost = cost + (ts[k][:, j:j + 1] - q4[:, j:j + 1] - lev[k][:, cand // 2 ** (4 - k)]) ** 2
        r = cost.argmin(1); R[:, j] = r
        if j + 1 < n:
            for k in KS:
                q = q4[:, j] + lev[k].gather(1, (r // 2 ** (4 - k))[:, None])[:, 0]
                ts[k][:, j + 1:] -= ((ts[k][:, j] - q) / U[j, j])[:, None] * U[j, j + 1:][None]
    return R


def refit(W, q4, R, lev):
    new = {}
    for k in KS:
        p = R // 2 ** (4 - k)
        s_ = torch.zeros_like(lev[k]).scatter_add_(1, p, W - q4); c = torch.zeros_like(lev[k]).scatter_add_(1, p, torch.ones_like(W))
        new[k] = torch.where(c > 0, s_ / c.clamp_min(1), lev[k])
    return new


@torch.no_grad()
def encode_residual(W, U, q4):
    """W (rows, cols), U the damped upper Cholesky factor of H^-1, q4 the 4-bit reconstruction. Returns the residual indices
    (rows, cols) in 0..15 and the per-row offsets (rows, 30) fp16, ordered k = 1, 2, 3, 4."""
    r_ = (W - q4).abs()
    m = torch.kthvalue(r_, max(1, int(QUANTILE * W.shape[1])), dim=1).values[:, None].clamp_min(1e-9)
    lev = mid_levels(m); R = choose(W, U, q4, lev)
    lev = refit(W, q4, R, lev); R = choose(W, U, q4, lev)
    return R, torch.cat([lev[k] for k in KS], 1).half()
