"""Qwen3-30B-A3B-Instruct-2507 with Any-Rate experts for the demo: the experts stay on the NVMe (files of moe_pack.py) and a VRAM cache of a
fixed size in GB holds the ones in use. A cached expert keeps the planes it was read with: a higher mode reads only the planes it lacks
(the modes are nested), a lower one only its small table; the least recently used experts leave the cache when it is full. A missing piece
is read with O_DIRECT (no page cache, so host DRAM is not a second cache) into pinned memory and copied to VRAM, the experts of a layer in
parallel. The non-expert weights (attention, router, norms, embeddings, lm_head) stay in VRAM in bf16; the experts are computed by the
grouped kernels of kernels/csrc/ar_moe.cu (ladder: ar_moe, 5-8 bits: + ar_res_moe). Decoding runs the transformers model eagerly, one token
per step (the prompt too)."""
import collections, ctypes, os, threading, time
from concurrent.futures import ThreadPoolExecutor
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, DynamicCache
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeRotaryEmbedding

BITS = [2, 2.5, 3, 3.5, 4, 5, 6, 7, 8]; K_OF = {2: 3, 2.5: 4, 3: 5, 3.5: 6, 4: 7}
PAGE = 4096; align = lambda n: (n + PAGE - 1) // PAGE * PAGE
LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernels", "lib", "libar_moe.so")


def mode(b):
    """(ladder planes K, residual planes r) of a bit width"""
    return (K_OF[b], 0) if b <= 4 else (7, int(b - 4))


class Layout:
    """byte sizes and offsets of one expert in the files of moe_pack.py (gate, up: I x H; down: H x I)"""
    def __init__(self, I, H):
        n = I * H; self.usz, self.hsz = n // 8, n // 16; self.rows, self.cols = (I, I, H), (H, H, I)
        self.planes = {K: 3 * self.usz + 3 * (K - 1) * self.hsz for K in range(3, 8)}          # plane prefix of mode K (page multiples)
        self.plane_slot = align(self.planes[7])
        tb = lambda K: [r * ((1 << K) * 2 if K <= 6 else 132) for r in self.rows]
        self.tab = {K: sum(tb(K)) for K in range(3, 8)}; self.tab_slot = {K: align(v) for K, v in self.tab.items()}
        self.tab_off = {K: [sum(tb(K)[:m]) for m in range(3)] for K in range(3, 8)}
        self.res = {r: 3 * r * self.usz for r in range(0, 5)}; self.res_slot = align(self.res[4])
        lv = lambda r: [rows * (1 << r) * 2 for rows in self.rows]
        self.lev = {r: sum(lv(r)) for r in range(1, 5)}; self.lev_slot = {r: align(v) for r, v in self.lev.items()}
        self.lev_off = {r: [sum(lv(r)[:m]) for m in range(3)] for r in range(1, 5)}
        self.poff = [[m * self.usz if p == 0 else 3 * self.usz + 3 * (p - 1) * self.hsz + m * self.hsz for p in range(7)] for m in range(3)]

    def expert_bytes(self, b):
        K, r = mode(b); return self.planes[K] + self.tab[K] + (self.res[r] + self.lev[r] if r else 0)


class Entry:
    __slots__ = ("K", "pl", "tabs", "r", "res", "levs")
    def __init__(self): self.K, self.pl, self.tabs, self.r, self.res, self.levs = 0, None, {}, 0, None, {}
    def nbytes(self): return sum(t.numel() for t in [self.pl, self.res, *self.tabs.values(), *self.levs.values()] if t is not None)


class Staging:
    def __init__(self, nbytes):
        self.raw = torch.empty(nbytes + PAGE, dtype=torch.uint8, pin_memory=True); off = (-self.raw.data_ptr()) % PAGE
        self.buf = self.raw[off:off + nbytes]; self.mv = memoryview(self.buf.numpy()); self.stream = torch.cuda.Stream()


class ExpertCache:
    """per layer an LRU of experts within budget / layers bytes; counters: ssd_bytes, hit (all planes of the mode cached), miss"""
    def __init__(self, d, lay, L, budget, workers=8):
        self.lay, self.L, self.budget = lay, L, budget; self.lru = [collections.OrderedDict() for _ in range(L)]; self.used = [0] * L
        fd = lambda n: os.open(os.path.join(d, n), os.O_RDONLY | os.O_DIRECT)
        self.has_res = os.path.exists(os.path.join(d, "L0.res"))         # without the residual files: 2-4 bits only
        self.f = [{"planes": fd(f"L{li}.planes"), **{f"tab{K}": fd(f"L{li}.tab{K}") for K in range(3, 8)},
                   **({"res": fd(f"L{li}.res"), **{f"lev{r}": fd(f"L{li}.lev{r}") for r in range(1, 5)}} if self.has_res else {})} for li in range(L)]
        self.tls = threading.local(); self.ex = ThreadPoolExecutor(workers); self.lock = threading.Lock(); self.counters = collections.Counter()

    def _read(self, fd, off, n, dst):
        """n bytes at off (both page multiples) of a file into the GPU tensor dst"""
        if not hasattr(self.tls, "st"): self.tls.st = Staging(max(self.lay.plane_slot, self.lay.res_slot))
        st = self.tls.st; assert os.preadv(fd, [st.mv[:n]], off) == n
        with torch.cuda.stream(st.stream): dst.copy_(st.buf[:n], non_blocking=True)
        st.stream.synchronize(); return n

    def _fill(self, li, e, ent, K, r):
        lay, f, got = self.lay, self.f[li], 0
        if ent.K < K:                                                    # the planes this mode adds (all of them for a new expert)
            new = torch.empty(lay.planes[K], dtype=torch.uint8, device="cuda"); a = lay.planes[ent.K] if ent.K else 0
            if a: new[:a].copy_(ent.pl)                                     # on the compute stream, ordered before the layer's kernels
            got += self._read(f["planes"], e * lay.plane_slot + a, lay.planes[K] - a, new[a:]); ent.pl, ent.K = new, K
        if K not in ent.tabs:
            t = torch.empty(lay.tab_slot[K], dtype=torch.uint8, device="cuda"); got += self._read(f[f"tab{K}"], e * lay.tab_slot[K], lay.tab_slot[K], t); ent.tabs[K] = t
        if ent.r < r:
            new = torch.empty(lay.res[r], dtype=torch.uint8, device="cuda"); a = lay.res[ent.r]
            if a: new[:a].copy_(ent.res)
            got += self._read(f["res"], e * lay.res_slot + a, lay.res[r] - a, new[a:]); ent.res, ent.r = new, r
        if r and r not in ent.levs:
            t = torch.empty(lay.lev_slot[r], dtype=torch.uint8, device="cuda"); got += self._read(f[f"lev{r}"], e * lay.lev_slot[r], lay.lev_slot[r], t); ent.levs[r] = t
        return got

    def get(self, li, ids, K, r):
        with self.lock:
            lru = self.lru[li]; ents, jobs = [], []
            for e in ids:
                ent = lru.get(e)
                if ent is None: ent = lru[e] = Entry()
                lru.move_to_end(e); ents.append(ent)
                if ent.K >= K and ent.r >= r: self.counters["hit"] += 1
                else: self.counters["miss"] += 1
                if ent.K < K or K not in ent.tabs or ent.r < r or (r and r not in ent.levs):
                    self.used[li] -= ent.nbytes(); jobs.append((e, ent))
            if jobs: torch.cuda.current_stream().synchronize()               # freed slots may still be read by queued kernels
            done = [self.ex.submit(self._fill, li, e, ent, K, r) for e, ent in jobs]
            for j in done: self.counters["ssd_bytes"] += j.result()
            for e, ent in jobs: self.used[li] += ent.nbytes()
            self._evict(li, keep=set(ids)); return ents

    def _evict(self, li, keep=()):
        lru, cap = self.lru[li], self.budget / self.L
        for e in list(lru):
            if self.used[li] <= cap: break
            if e not in keep: self.used[li] -= lru.pop(e).nbytes()

    def resize(self, budget):
        with self.lock:
            self.budget = budget
            for li in range(self.L): self._evict(li)

    def close(self):                                                # before the model is unloaded: the reader threads and the files
        self.ex.shutdown(wait=True)
        for f in self.f:
            for fd in f.values(): os.close(fd)
        self.lru = [collections.OrderedDict() for _ in range(self.L)]

    def info(self):
        return {"budget_gb": self.budget / 1e9, "used_gb": sum(self.used) / 1e9, "experts": sum(len(x) for x in self.lru)}


class Block(nn.Module):
    """the sparse MoE block of one decode token: router in VRAM, the selected experts from the cache"""
    def __init__(self, cfg, li, eng):
        super().__init__(); self.li, self.eng, self.k, self.norm = li, eng, cfg.num_experts_per_tok, cfg.norm_topk_prob
        self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)

    def forward(self, h):
        b, t, d = h.shape; x = h.view(-1, d); assert x.shape[0] == 1, "one token per step"
        p = F.softmax(self.gate(x), dim=-1, dtype=torch.float); w, ids = torch.topk(p, self.k, dim=-1)
        if self.norm: w = w / w.sum(-1, keepdim=True)
        K, r = mode(self.eng.bits); ents = self.eng.cache.get(self.li, ids[0].tolist(), K, r)
        return self.eng.experts(x.contiguous(), w.to(x.dtype)[0], ents, K, r).view(b, t, d), None


class MoEEngine:
    def __init__(self, d, budget_gb=6.0, lmax=4096, log=print):
        self.dir, self.lmax = d, lmax; cfg = self.cfg = AutoConfig.from_pretrained(d); self.L = cfg.num_hidden_layers
        self.lay = Layout(cfg.moe_intermediate_size, cfg.hidden_size); self.bits = 4.0
        self.lib = ctypes.CDLL(LIB); self.lib.ar_moe.restype = self.lib.ar_res_moe.restype = ctypes.c_int
        self.cache = ExpertCache(d, self.lay, self.L, budget_gb * 1e9); self.modes = BITS if self.cache.has_res else BITS[:5]
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"): model = Qwen3MoeForCausalLM(cfg)
        for li, layer in enumerate(model.model.layers): layer.mlp = Block(cfg, li, self)
        model.to_empty(device="cuda"); sd = {}
        with safe_open(os.path.join(d, "nonexpert.safetensors"), "pt") as fh: sd = {k: fh.get_tensor(k) for k in fh.keys()}
        self.nonexpert_bytes = sum(t.numel() * t.element_size() for t in sd.values()); self.embed_bytes = sd["model.embed_tokens.weight"].numel() * 2
        missing, unexpected = model.load_state_dict(sd, strict=False); assert not missing and not unexpected, (missing[:3], unexpected[:3]); del sd
        model.model.rotary_emb = Qwen3MoeRotaryEmbedding(cfg, device="cuda"); torch.set_default_dtype(torch.float32)
        self.model = model.eval(); self.tok = AutoTokenizer.from_pretrained(d); self.kv = DynamicCache(); self.ctx = []
        eos = cfg.eos_token_id; self.eos = set(eos if isinstance(eos, list) else [eos]) | ({self.tok.eos_token_id} if self.tok.eos_token_id is not None else set())
        log(f"Qwen3 MoE: {self.L} layers x {cfg.num_experts} experts on the NVMe, expert cache {budget_gb:g} GB, VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GB")

    # ---- the experts of one layer: gate/up (2k matrices), activation, down (k matrices), weighted sum
    def _call(self, fn, mats, k, *extra):
        n = len(mats); c = ctypes.c_void_p
        return fn(*[(c * len(v))(*v) if t == "p" else (ctypes.c_int * n)(*v) if t == "i" else (ctypes.c_longlong * n)(*v) for t, v in extra], n, k,
                  c(torch.cuda.current_stream().cuda_stream))

    def _ladder(self, ents, ms, xs, ys, K):
        lay = self.lay; P, T = [], []
        for ent in ents:
            b = ent.pl.data_ptr(); tb = ent.tabs[K].data_ptr()
            for m in ms: P += [b + lay.poff[m][p] if p < K else 0 for p in range(7)]; T.append(tb + lay.tab_off[K][m])
        rows = [lay.rows[m] for _ in ents for m in ms]; cols = [lay.cols[m] for _ in ents for m in ms]
        rc = self._call(self.lib.ar_moe, rows, K, ("p", P), ("p", T), ("p", xs), ("p", ys), ("i", rows), ("i", cols)); assert rc == 0, f"ar_moe error {rc}"

    def _residual(self, ents, ms, xs, ys, r):
        lay = self.lay; P, S, V = [], [], []
        for ent in ents:
            for m in ms: P.append(ent.res.data_ptr() + m * lay.usz); S.append(3 * lay.usz); V.append(ent.levs[r].data_ptr() + lay.lev_off[r][m])
        rows = [lay.rows[m] for _ in ents for m in ms]; cols = [lay.cols[m] for _ in ents for m in ms]
        rc = self._call(self.lib.ar_res_moe, rows, r, ("p", P), ("l", S), ("p", V), ("p", xs), ("p", ys), ("i", rows), ("i", cols)); assert rc == 0, f"ar_res_moe error {rc}"

    def experts(self, x, w, ents, K, r):
        k, I, H = len(ents), self.lay.rows[0], self.lay.cols[0]
        gu = torch.empty(2 * k, I, dtype=x.dtype, device=x.device); xs = [x.data_ptr()] * (2 * k); ys = [gu[i].data_ptr() for i in range(2 * k)]
        self._ladder(ents, (0, 1), xs, ys, K)
        if r: self._residual(ents, (0, 1), xs, ys, r)
        act = (F.silu(gu[0::2]) * gu[1::2]).contiguous(); yd = torch.empty(k, H, dtype=x.dtype, device=x.device)
        xs = [act[j].data_ptr() for j in range(k)]; ys = [yd[j].data_ptr() for j in range(k)]
        self._ladder(ents, (2,), xs, ys, K)
        if r: self._residual(ents, (2,), xs, ys, r)
        return (w[:, None] * yd).sum(0, keepdim=True)

    # ---- sizes
    def params(self):
        c = self.cfg; return self.L * c.num_experts * 3 * c.moe_intermediate_size * c.hidden_size + self.nonexpert_bytes // 2

    def bytes_per_token(self, b):
        """weights one decoded token uses: its 8 experts in every layer (mode b) and the non-expert weights but the embedding table"""
        return self.L * self.cfg.num_experts_per_tok * self.lay.expert_bytes(b) + self.nonexpert_bytes - self.embed_bytes

    def stored_bytes(self):
        return sum(os.path.getsize(os.path.join(self.dir, f)) for f in os.listdir(self.dir) if f.startswith("L") or f == "nonexpert.safetensors")

    # ---- decoding
    def _forward(self, token, b):
        p = len(self.ctx); assert p < self.lmax, "context full"; self.bits = b
        out = self.model(input_ids=torch.tensor([[token]], device="cuda"), past_key_values=self.kv, use_cache=True, cache_position=torch.tensor([p], device="cuda"))
        self.logits = out.logits[:, -1]; self.ctx.append(token)

    @torch.inference_mode()
    def generate(self, ids, bits, max_new, on_token, greedy=False, temperature=0.7, top_p=0.9, stop=lambda: False):
        """as Engine.generate (engine.py): the cached context is reused when `ids` extends it; bits() is read before every token"""
        n = 0
        while n < len(self.ctx) and n < len(ids) and self.ctx[n] == ids[n]: n += 1
        if n == len(ids): n -= 1
        if n == 0: self.kv = DynamicCache()
        elif n < len(self.ctx): self.kv.crop(n)
        self.ctx = self.ctx[:n]; torch.cuda.synchronize(); t0 = time.time()
        for t in ids[n:]: self._forward(t, bits())
        torch.cuda.synchronize(); t1 = time.time(); out, used, prev = [], [], ""
        for _ in range(max_new):
            if stop(): break
            lg = self.logits[0].float()
            if greedy: nxt = int(lg.argmax())
            else:
                pr = torch.softmax(lg / temperature, -1); sp, si = pr.sort(descending=True); keep = sp.cumsum(0) - sp < top_p
                nxt = int(si[keep][torch.multinomial(sp[keep] / sp[keep].sum(), 1)])
            if nxt in self.eos: break
            b = bits(); out.append(nxt); used.append(b); text = self.tok.decode(out, skip_special_tokens=True)
            if not text.endswith("\ufffd"): on_token(text[len(prev):], b); prev = text      # a character split over tokens: wait for the rest
            self._forward(nxt, b)
        torch.cuda.synchronize(); t2 = time.time()
        return {"prompt_tokens": len(ids) - n, "context_tokens": len(self.ctx), "kv_reused": n > 0, "prefill_s": t1 - t0, "decode_s": t2 - t1,
                "tokens": len(out), "tokens_per_s": len(out) / (t2 - t1) if out else 0.0, "bits_used": sorted(set(used))}
