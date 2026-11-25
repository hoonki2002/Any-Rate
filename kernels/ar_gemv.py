"""Batch-1 decode GEMV of an Any-Rate layer (UHHHHHH) read directly from its checkpoint tensors (ctypes wrapper of csrc/, see build.sh).

  m = ARGemv(t)                  t: the tensors of one layer of the checkpoint ('plane0'.., 'table_codes', 'table_lo', 'table_step';
                                 'res_planes', 'res_lev' for the 5-8-bit modes)
  y = m(x, k, v14=True, W=16)    x: (1, cols) fp16; k: planes read (3..7 = 2, 2.5, 3, 3.5, 4 bits); one kernel and its launch config
  m.residual(x, y, r)            5-8 bits (4 + r): adds the first r residual planes to the 4-bit output y (csrc/ar_residual.cu)

The planes are rearranged once into the layout of the kernels (same bits), and the fp16 table of every mode k < 7 is precomputed (2^k
values per row, the mean of the 2^(7-k) children of the stored 8-bit table, as any_rate/format.py); mode 7 builds its table in the
kernel from the 8-bit codes. Kernels (one lane per row, conflict-free shared tables):
  v8   tables addressed with byte_perm, split-K over S blocks           v11  v8 with the Hamming decode in the message domain
  v12  v11 as a persistent stream-K kernel                              v14  v12 with cp.async prefetch of the next tile's table
  dense=True (v11, v14): a dense 64-entry pair table at k = 3.  The benchmark (bench.py) picks the fastest kernel and config per shape and mode."""
import ctypes, os
import torch

LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
_libs = {}


def _lib(name, fn):
    if name not in _libs:
        path = os.path.join(LIB, f"lib{name}.so")
        if not os.path.exists(path): raise RuntimeError(f"{path} is missing: run `bash kernels/build.sh`")
        _libs[name] = ctypes.CDLL(path)
    f = getattr(_libs[name], fn); f.restype = ctypes.c_int; return f


def _pad(t, rows):
    return torch.cat([t, torch.zeros(rows - t.shape[0], *t.shape[1:], dtype=t.dtype, device=t.device)]) if t.shape[0] < rows else t


class ARGemv:
    def __init__(self, t, dev="cuda"):
        P = {k: v.to(dev) for k, v in t.items()}; NSM = torch.cuda.get_device_properties(dev).multi_processor_count
        codes = P["table_codes"]; self.rows, self.cols = codes.shape[0], P["plane0"].shape[1] * 8; self.nsm = NSM
        R, C = self.rows, self.cols; Rp = (R + 31) // 32 * 32; T = Rp // 32; nch = C // 64; self.rows_pad = Rp
        self.U = _pad(P["plane0"], Rp).view(T, 32, nch, 8).permute(0, 2, 1, 3).contiguous()
        self.H = [_pad(P[f"plane{p}"], Rp).view(T, 32, nch, 4).permute(0, 2, 1, 3).contiguous() for p in range(1, 7)]
        lo = P["table_lo"].half().reshape(-1); st = P["table_step"].half().reshape(-1)
        self.lo = _pad(lo, Rp).contiguous(); self.st = _pad(st, Rp).contiguous()
        self.codes = _pad(codes, Rp).view(T, 32, 32, 4).permute(0, 2, 1, 3).contiguous()          # [T][j/4][lane][4]
        traw = lo.float()[:, None] + codes.float() * st.float()[:, None]; self.tab = {}
        for k in range(3, 7):
            g = 1 << (7 - k); self.tab[k] = _pad((traw.view(R, 1 << k, g).sum(-1) / g).half(), Rp).view(T, 32, 1 << k).permute(0, 2, 1).contiguous()
        self.dummy = torch.zeros(16, dtype=torch.uint8, device=dev); self.Harr = (ctypes.c_void_p * 6)(*[h.data_ptr() for h in self.H])
        self.ws = {}; self.cnt = torch.zeros(T, dtype=torch.int32, device=dev); self.U11 = None; self.ws12 = None
        self.res = (P["res_planes"].contiguous(), P["res_lev"].half().contiguous()) if "res_planes" in P else None

    def _u_phase(self):
        """phase-ordered U words of v11-v14 (same bits): word h, bit 8j + 4p + 2(g&1) + x = U of block 2g+p, column 4x+j, g = 2h + ((bit>>1)&1)"""
        if self.U11 is None:
            dev = self.U.device; bits = ((self.U.long()[..., None] >> torch.arange(8, device=dev)) & 1)
            pos = torch.arange(32, device=dev); j = pos >> 3; p = (pos >> 2) & 1; o = pos & 3; words = []
            for h in range(2):
                g = 2 * h + (o >> 1); words.append((bits[..., 2 * g + p, 4 * (o & 1) + j] << pos).sum(-1))
            w = torch.stack(words, -1); self.U11 = torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32).contiguous()
        return self.U11

    def _args(self, U, x, y, ws, k):
        tab = self.tab[k].data_ptr() if k != 7 else self.dummy.data_ptr(); c = ctypes.c_void_p
        return [c(U.data_ptr()), self.Harr, c(tab), c(self.codes.data_ptr()), c(self.lo.data_ptr()), c(self.st.data_ptr()), c(x.data_ptr()),
                c(y.data_ptr()), c(ws), c(self.cnt.data_ptr()), self.rows, self.cols, k]

    def __call__(self, x, k, S=0, W=16, v8=False, v11=False, v12=False, v14=False, dense=False):
        assert x.dtype == torch.float16 and x.shape == (1, self.cols) and x.is_contiguous()
        y = torch.empty(1, self.rows, dtype=x.dtype, device=x.device); stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        if v12 or v14:                                   # persistent stream-K: S = number of blocks (0: as many as fit)
            if self.ws12 is None: self.ws12 = torch.zeros(self.rows_pad, dtype=torch.float32, device=x.device)
            f = _lib("ar_gemv_v14_dense" if v14 and dense else "ar_gemv_v14" if v14 else "ar_gemv_v12", "ar14_gemv" if v14 else "ar12_gemv")
            r = f(*self._args(self._u_phase(), x, y, self.ws12.data_ptr(), k), S, W, stream)
        else:                                            # v8 / v11: split-K over S blocks
            nch = self.cols // 64; S = S or 1
            tb = ((120 if k == 3 else 256) * 256 if k <= 4 else (1 << k) * 256) if v8 else (120 if k == 3 else 1 << k) * 256   # table bytes
            while tb + W * 128 + ((nch + S - 1) // S) * 256 > 99 * 1024: S += 1                 # shared memory limit
            key = (S, 1)
            if S > 1 and key not in self.ws: self.ws[key] = torch.empty(S * self.rows_pad, dtype=torch.float32, device=x.device)
            ws = self.ws[key].data_ptr() if S > 1 else 0
            if v8:
                r = _lib("ar_gemv_v8", "ar8_gemv")(*self._args(self.U, x, y, ws, k), 1, S, W, 0, 1, stream)
            else:
                r = _lib("ar_gemv_v11_dense" if dense else "ar_gemv_v11", "ar11_gemv")(*self._args(self._u_phase(), x, y, ws, k), S, W, stream)
        assert r == 0, f"AR GEMV error {r}"; return y

    def residual(self, x, y, r):
        """y += the residual of the first r residual planes (y: the 4-bit output of this layer), in place"""
        assert self.res is not None, "the residual planes were not loaded"
        rc = _lib("ar_residual", "ar_res_gemv")(ctypes.c_void_p(self.res[0].data_ptr()), ctypes.c_void_p(self.res[1].data_ptr()), ctypes.c_void_p(x.data_ptr()),
                                                ctypes.c_void_p(y.data_ptr()), self.rows, self.cols, r, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        assert rc == 0, f"AR residual error {rc}"; return y
