"""Nested ladder of coded index planes: encoding of one weight matrix.

Each weight stores a b-bit index into a per-row table. The index bits are planes, the most significant first. A layout names
the type of every plane: 'U' stores one free bit per weight, 'H' constrains the 8 bits of a block of 8 columns to a codeword of
the extended Hamming [8,4,4] code (4 bits per block). The mode that reads the first k planes looks up a table T_k, the mean of the
2^(b-k) children of the stored top-mode table T. Encoding (Appendix D of the paper):
  - columns are processed left to right in blocks of 8 with GPTQ-style error feedback, one target per mode (per-mode targets);
  - in each block, the codewords of the H planes are chosen by a prefix beam search (B partial candidates per plane), the K best
    full candidates are evaluated exactly, and the U bits are chosen per column;
  - after each pass the shared table is refit in closed form (weighted least squares over all modes) and stored in 8 bits;
  - N_ref = 4 rounds; the round with the lowest weighted proxy loss is kept.
"""
import itertools
import torch


def hamming8():
    """The 16 codewords of the extended Hamming [8,4,4] code as +-1 vectors (bit 1 <=> -1); row a = message a (4 bits, a_0 first)."""
    G = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.long)
    U = torch.tensor(list(itertools.product([0, 1], repeat=4)), dtype=torch.long)
    return (1 - 2 * ((U @ G) % 2)).float()


class Layout:
    """Plane layout such as 'UHHHHHH'. idx[a, f] is the full index for coded-bit combination a and free-bit pattern f.
    comb (T, 8): the coded-bit combination of each column for every candidate (one codeword per H plane), built only for full
    enumeration (beam = 0)."""
    def __init__(self, name, dev, enumerate_all=False):
        assert set(name) <= {"U", "H"}, name
        self.name, self.b = name, len(name)
        self.fi = [i for i, c in enumerate(name) if c == "U"]
        self.ci = [i for i, c in enumerate(name) if c == "H"]
        self.L = 2 ** self.b
        m = len(self.ci)
        free_bits = torch.tensor(list(itertools.product([0, 1], repeat=len(self.fi))), device=dev)
        cbits = torch.tensor(list(itertools.product([0, 1], repeat=m)), device=dev) if m else torch.zeros(1, 0, dtype=torch.long, device=dev)
        idx = torch.zeros(len(cbits), len(free_bits), dtype=torch.long, device=dev)
        for q, p in enumerate(self.ci):
            idx += cbits[:, q:q + 1] << (self.b - 1 - p)
        for q, p in enumerate(self.fi):
            idx += free_bits[None, :, q] << (self.b - 1 - p)
        self.idx = idx                                                  # (A, F)
        C = hamming8().to(dev)
        self.sgn = [(C < 0).long() for _ in self.ci]                    # per H plane: (16, 8) codeword bit of each column
        self.T = 16 ** m
        if enumerate_all and m:
            grids = torch.meshgrid(*[torch.arange(16, device=dev) for _ in self.ci], indexing='ij')
            comb = torch.zeros(self.T, 8, dtype=torch.long, device=dev)
            for q, t in enumerate(g.reshape(-1) for g in grids):
                comb += (C[t] < 0).long() << (m - 1 - q)
            self.comb = comb
            A = 2 ** m
            if self.T * 8 * A <= 2 ** 28:                                # ranking matrix (8A, T); larger layouts rank by a chunked gather-sum
                onehot = torch.zeros(self.T, 8 * A, device=dev)
                onehot.scatter_(1, comb + A * torch.arange(8, device=dev)[None], 1.0)
                self.onehot = onehot.T.contiguous()
            else:
                self.onehot = None
        elif enumerate_all:
            self.comb, self.T = torch.zeros(1, 8, dtype=torch.long, device=dev), 1


def prep(H, damp):
    """Damped H (damp x mean diagonal; zero diagonal entries set to 1) -> upper Cholesky factor U of H^-1, as in GPTQ."""
    Hd = H.double().clone()
    dead = Hd.diagonal() == 0
    Hd[dead, dead] = 1.0
    Hd.diagonal().add_(damp * Hd.diagonal().mean())
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hd)), upper=True)
    return U.float()


def proxy(W, Wq, H):
    E = W - Wq
    return float(((E @ H) * E).sum())


def init_table(W, H, b, iters=15):
    """Per-row k-means of the row's weights weighted by diag(H), quantile initialisation; 2^b sorted levels per row."""
    hd = H.diagonal().clamp_min(1e-12)
    L = 2 ** b
    q = torch.linspace(0.5 / L, 1 - 0.5 / L, L, device=W.device)
    lut = torch.quantile(W, q, dim=1).T.contiguous()
    for _ in range(iters):
        mid = ((lut[:, 1:] + lut[:, :-1]) / 2).contiguous()
        a = torch.searchsorted(mid, W.contiguous())
        wgt = hd.expand_as(W)
        num = torch.zeros_like(lut).scatter_add_(1, a, W * wgt)
        den = torch.zeros_like(lut).scatter_add_(1, a, wgt)
        lut = torch.where(den > 0, num / den.clamp_min(1e-20), lut)
        lut, _ = lut.sort(1)
    return lut


@torch.no_grad()
def prefix_beam(ts, Ub, luts, lay, modes, w, B, K):
    """Prefix beam over the codewords of the H planes. Mode m reads its first k_m planes, so its cost (without in-block feedback, best U
    bit per column) is fixed once the codewords of the H planes it reads are chosen. Keeps the B best partial candidates per H plane and
    returns the K best full candidates as coded-bit combinations per column, (R, K, 8)."""
    R = ts[0].shape[0]; dev = ts[0].device
    m = len(lay.ci); b = lay.b
    dw = 1.0 / Ub.diagonal() ** 2
    ncod = [sum(1 for p in lay.ci if p < k) for k in modes]
    beams = None
    for q in range(1, m + 1):
        inc = [mi for mi in range(len(modes)) if ncod[mi] <= q]
        a_full = torch.arange(2 ** q, device=dev) << (m - q)
        idx = lay.idx[a_full]
        cost_tab = None
        if inc:
            c = 0
            for mi in inc:
                v = luts[mi][:, idx >> (b - modes[mi])]
                c = c + w[mi] * (ts[mi][:, :, None, None] - v[:, None]) ** 2
            cost_tab = c.min(-1).values * dw[None, :, None]            # (R, 8, 2^q)
        S = lay.sgn[q - 1]; nw = S.shape[0]
        if beams is None:
            prev = torch.zeros(R, 1, 8, dtype=torch.long, device=dev); pc = torch.zeros(R, 1, device=dev)
        else:
            prev, pc = beams
        nbm = prev.shape[1]
        a_new = ((prev[:, :, None, :] << 1) | S[None, None, :, :]).reshape(R, nbm * nw, 8)
        if cost_tab is None:
            cost = pc[:, :, None].expand(R, nbm, nw).reshape(R, -1)
        else:
            gidx = (torch.arange(8, device=dev)[None, None, :] * (2 ** q) + a_new).reshape(R, -1)
            cost = cost_tab.reshape(R, -1).gather(1, gidx).reshape(R, nbm * nw, 8).sum(-1)
        keep = min(B if q < m else K, cost.shape[1])
        top = cost.topk(keep, dim=1, largest=False).indices
        beams = (a_new.gather(1, top[:, :, None].expand(-1, -1, 8)), cost.gather(1, top))
    return beams[0]


def rank_all(ts, Ub, vals, lay, modes, w, K):
    """Full enumeration: rank all 16^n_H candidates by the cost without in-block feedback and return the K best, (R, K, 8)."""
    R = ts[0].shape[0]
    dw = 1.0 / Ub.diagonal() ** 2
    c = 0
    for m in range(len(modes)):
        c = c + w[m] * (ts[m][:, :, None, None] - vals[m][:, None]) ** 2
    c1 = c.min(-1).values * dw[None, :, None]                          # (R, 8, A)
    if lay.onehot is not None:
        sel = (c1.reshape(R, -1) @ lay.onehot).topk(K, dim=1, largest=False).indices
    else:
        flat = c1.reshape(R, -1); offs = lay.idx.shape[0] * torch.arange(8, device=c1.device)[None]
        ch = max(1, int(2.5e8 // (R * 8))); bv = bi = None
        for s0 in range(0, lay.T, ch):
            sc = flat[:, lay.comb[s0:s0 + ch] + offs].sum(-1)
            if bv is not None: sc = torch.cat([bv, sc], 1)
            ii = torch.arange(s0, min(s0 + ch, lay.T), device=c1.device)[None].expand(R, -1)
            if bi is not None: ii = torch.cat([bi, ii], 1)
            v, k = sc.topk(min(K, sc.shape[1]), dim=1, largest=False); bv, bi = v, ii.gather(1, k)
        sel = bi
    return lay.comb[sel]


def block_step(ts, Ub, luts, lay, modes, w, B, K):
    """One block of 8 columns: ts are the per-mode targets (R, 8), luts the per-mode tables (R, 2^k). Mode k reads index >> (b - k).
    Candidates: prefix beam of width B (B > 0); full enumeration ranked to the K best (B = 0, K > 0); every candidate (B = K = 0).
    Returns the per-mode errors E (R, 8), already divided by the diagonal of Ub, and the chosen full indices (R, 8)."""
    R = ts[0].shape[0]
    A, Fp = lay.idx.shape
    vals = [lut[:, lay.idx >> (lay.b - k)] for lut, k in zip(luts, modes)]
    if not lay.ci:
        combs = torch.zeros(R, 1, 8, dtype=torch.long, device=ts[0].device)
    elif B:
        combs = prefix_beam(ts, Ub, luts, lay, modes, w, B, min(K, B))
    elif K and K < lay.T:
        combs = rank_all(ts, Ub, vals, lay, modes, w, K)
    else:
        combs = lay.comb[None].expand(R, -1, -1)
    T = combs.shape[1]
    Es = [[] for _ in modes]
    Is = []
    cost = ts[0].new_zeros(R, T)
    for i in range(8):
        cands, tis, tot = [], [], 0
        for m in range(len(modes)):
            ti = ts[m][:, i:i + 1].expand(R, T)
            for k in range(i):
                ti = ti - Es[m][k] * Ub[k, i]
            cand = vals[m].gather(1, combs[:, :, i:i + 1].expand(-1, -1, Fp))
            cands.append(cand)
            tis.append(ti)
            tot = tot + w[m] * (ti[..., None] - cand) ** 2
        j = tot.argmin(-1)
        for m in range(len(modes)):
            e = (tis[m] - cands[m].gather(2, j[..., None])[..., 0]) / Ub[i, i]
            Es[m].append(e)
            cost = cost + w[m] * e * e
        Is.append(lay.idx[combs[:, :, i], j])
    best = cost.argmin(1, keepdim=True)
    E = [torch.cat([e.gather(1, best) for e in Es[m]], 1) for m in range(len(modes))]
    I = torch.cat([x.gather(1, best) for x in Is], 1)
    return E, I


def refit_table(W, H, I, b, modes, w, chunk=256, budget=3e9):
    """Shared table: one stored table T per row (2^b entries); mode k reads T_k = T averaged over its 2^(b-k) children. Minimises
    sum_m w_m (w - P B_m T)^T H (w - P B_m T) over T for fixed indices (normal equations, ridge 1e-6 x mean diagonal)."""
    rows, cols = W.shape
    L = 2 ** b
    dev = W.device
    Bs = []
    for k in modes:
        g = 2 ** (b - k)
        par = torch.arange(L, device=dev) // g
        Bs.append((par[:, None] == par[None, :]).float() / g)
    chunk = max(8, min(chunk, int(budget // (cols * L))))
    out = torch.empty(rows, L, device=dev)
    for a in range(0, rows, chunk):
        P = torch.nn.functional.one_hot(I[a:a + chunk], L).float()
        HP = torch.einsum('ij,rjl->ril', H, P)
        G = torch.einsum('ril,rim->rlm', P, HP)
        h = torch.einsum('ril,ri->rl', HP, W[a:a + chunk])
        M = sum(wm * B.T @ G @ B for wm, B in zip(w, Bs))
        rhs = sum(wm * h @ B for wm, B in zip(w, Bs))
        ridge = 1e-6 * M.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-12)
        M = M + ridge[:, None, None] * torch.eye(L, device=dev)
        out[a:a + chunk] = torch.linalg.solve(M, rhs)
    return out, Bs


def quantize_table_8bit(T):
    """8-bit table per row: lo and step = (max - min) / 255 rounded to fp16 first, then codes = round((T - lo) / step)."""
    lo, hi = T.min(1, keepdim=True).values, T.max(1, keepdim=True).values
    step = ((hi - lo) / 255).clamp_min(1e-12)
    lo, step = lo.half().float(), step.half().float().clamp_min(1e-12)
    code = torch.round((T - lo) / step).clamp(0, 255)
    return lo + code * step, (code.to(torch.uint8), lo.half(), step.half())


@torch.no_grad()
def encode_ladder(W, H, layout="UHHHHHH", first=3, weights=(1, 1, 2, 3, 4), damp=0.01, beam=64, K=64, target="per_mode", rounds=4, refit_budget=3e9):
    """Encode W (rows, cols) with Hessian H (cols, cols). Modes read first..b planes with weights `weights`.
    beam: prefix beam width B (0: full enumeration ranked to the K best; K = 0 as well: every candidate evaluated).
    target: 'per_mode' (one target per mode, the paper) or 'mean' (MatGPTQ-style mean propagation: the weighted mean of the mode errors
    is passed to the remaining columns of every mode).
    Returns (indices (rows, cols) long, 8-bit table (codes, lo, step), {k: reconstruction of mode k}, weighted relative proxy loss)."""
    assert target in ("per_mode", "mean"), target
    lay = Layout(layout, W.device, enumerate_all=not beam)
    modes = list(range(first, lay.b + 1))
    w = [float(v) for v in weights]
    w = [v / sum(w) for v in w]
    U = prep(H, damp)
    rows, cols = W.shape
    # nested initialisation (incremental upscaling): fit the lowest table, then split every entry into two children
    luts = [init_table(W, H, modes[0])]
    for k in modes[1:]:
        prev = luts[-1]
        spread = (prev[:, 1:] - prev[:, :-1]).abs().mean(1, keepdim=True) / 4 if prev.shape[1] > 1 else prev.abs().mean(1, keepdim=True)
        child = prev.repeat_interleave(2 ** (k - (luts[-1].shape[1].bit_length() - 1)), 1)
        offs = torch.linspace(-1, 1, child.shape[1] // prev.shape[1], device=W.device).repeat(prev.shape[1])[None] * spread
        luts.append(child + offs)
    base = float(((W @ H) * W).sum())
    best = None
    for _ in range(rounds):
        Wc = [W.clone() for _ in modes]
        I = torch.empty(rows, cols, dtype=torch.long, device=W.device)
        for b in range(0, cols, 8):
            E, idx = block_step([x[:, b:b + 8] for x in Wc], U[b:b + 8, b:b + 8], luts, lay, modes, w, beam, K)
            I[:, b:b + 8] = idx
            if b + 8 < cols:
                if target == "mean":
                    Em = sum(wm * Em_ for wm, Em_ in zip(w, E))
                    for m in range(len(modes)):
                        Wc[m][:, b + 8:] -= Em @ U[b:b + 8, b + 8:]
                else:
                    for m in range(len(modes)):
                        Wc[m][:, b + 8:] -= E[m] @ U[b:b + 8, b + 8:]
        T, Bs = refit_table(W, H, I, lay.b, modes, w, budget=refit_budget)
        T, q8 = quantize_table_8bit(T)
        luts = [(T @ B)[:, ::2 ** (lay.b - k)].contiguous() for B, k in zip(Bs, modes)]
        Ws = [lut.gather(1, I >> (lay.b - k)) for lut, k in zip(luts, modes)]
        obj = sum(a * p for a, p in zip(w, [proxy(W, x, H) / base for x in Ws]))
        if best is None or obj < best[0]:
            best = (obj, Ws, I.clone(), q8)
    return best[2], best[3], dict(zip(modes, best[1])), best[0]
