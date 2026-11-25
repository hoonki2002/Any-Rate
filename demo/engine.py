"""Batch-1 decoding of an Any-Rate checkpoint (Llama-3 / Qwen2 decoders) with the Any-Rate GEMV kernels (kernels/), one mode at a time.

Every linear layer stays in the checkpoint format in VRAM (bit planes, the 8-bit table and the residual planes); a mode only selects what
the kernels read: 2, 2.5, 3, 3.5, 4 bits read the first 3..7 planes, 5, 6, 7, 8 bits add the first 1..4 residual planes to 4 bits. One decode step per mode is
captured in a CUDA graph, so switching the mode between two tokens costs nothing. The prompt is processed one token at a time with the
same graph. Embeddings, norms, attention and the output head run in fp16; attention reads the K/V cache up to the smallest length bucket
(256, 512, ..., lmax) that holds the context, so there is one graph per mode and bucket."""
import glob, json, os, statistics, time
import torch
import torch.nn.functional as F
from safetensors import safe_open

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys; sys.path.insert(0, REPO)
from kernels.ar_gemv import ARGemv

BITS = [2, 2.5, 3, 3.5, 4, 5, 6, 7, 8]; LADDER = BITS[:5]; K_OF = {2: 3, 2.5: 4, 3: 5, 3.5: 6, 4: 7}       # ladder modes: planes read
PROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
CANDIDATES = ["v8 S=1 W=8", "v8 S=1 W=16", "v11d S=1 W=16", "v12 S=0 W=8", "v12 S=0 W=16", "v14 S=0 W=8", "v14 S=0 W=16", "v14d S=0 W=8"]


def parse(c):
    name, *kv = c.split(); kw = {a: int(b) for a, b in (t.split("=") for t in kv)}; kw[name.rstrip("d")] = True
    if name.endswith("d"): kw["dense"] = True
    return kw


def timed(calls, reps=10):
    """us per call: one CUDA graph over all calls, median of replays"""
    for _ in range(2):
        for f in calls: f()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in calls: f()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts) / len(calls) * 1e3


class Engine:
    def __init__(self, ckpt, lmax=4096, log=print):
        t0 = time.time(); self.ckpt, self.lmax, self.log = ckpt, lmax, log
        from transformers import AutoConfig, AutoTokenizer
        self.cfg = c = AutoConfig.from_pretrained(ckpt); self.tok = AutoTokenizer.from_pretrained(ckpt)
        assert c.model_type in ("llama", "qwen2"), f"{c.model_type}: Llama-3 and Qwen2 decoders only"
        self.D, self.NL, self.NH, self.NKV = c.hidden_size, c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads
        self.HD = getattr(c, "head_dim", None) or self.D // self.NH; self.eps = c.rms_norm_eps
        files = sorted(glob.glob(os.path.join(ckpt, "*.safetensors"))); where = {k: f for f in files for k in safe_open(f, "pt").keys()}
        get = lambda k: self._get(where, k)
        self.embed = get("model.embed_tokens.weight").half().cuda()
        self.lm_head = self.embed if "lm_head.weight" not in where else get("lm_head.weight").half().cuda()
        self.ln1 = [get(f"model.layers.{l}.input_layernorm.weight").half().cuda() for l in range(self.NL)]
        self.ln2 = [get(f"model.layers.{l}.post_attention_layernorm.weight").half().cuda() for l in range(self.NL)]
        self.lnf = get("model.norm.weight").half().cuda()
        self.bias = {(l, p): get(f"model.layers.{l}.{p}.bias").half().cuda() for l in range(self.NL) for p in PROJ if f"model.layers.{l}.{p}.bias" in where}
        self.lin = {}
        for l in range(self.NL):
            for p in PROJ:
                n = f"model.layers.{l}.{p}"; t = {f: get(f"{n}.ar.{f}").cuda() for f in ["plane0", *[f"plane{i}" for i in range(1, 7)], "table_codes", "table_lo", "table_step", "res_planes", "res_lev"]}
                self.lin[(l, p)] = ARGemv(t); del t
        torch.cuda.empty_cache()
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS          # the model's own RoPE (Llama-3 frequency scaling included)
        inv, scale = ROPE_INIT_FUNCTIONS[(getattr(c, "rope_scaling", None) or {}).get("rope_type", "default")](c, "cuda")
        f = torch.outer(torch.arange(lmax, device="cuda").float(), inv.float()); self.cos, self.sin = (f.cos() * scale).half(), (f.sin() * scale).half()
        self.kc = torch.zeros(self.NL, 1, self.NKV, lmax, self.HD, device="cuda").half(); self.vc = torch.zeros_like(self.kc)
        self.mask = torch.ones(lmax, lmax, dtype=torch.bool, device="cuda").tril()
        self.tokbuf = torch.zeros(1, dtype=torch.long, device="cuda"); self.pos = torch.zeros(1, dtype=torch.long, device="cuda")
        self.logits = torch.zeros(1, c.vocab_size, device="cuda").half()
        self.kcfg = self._kernel_configs(); self.graphs = {}; self.buckets = [n for n in (256, 512, 1024, 2048, 4096, 8192) if n < lmax] + [lmax]
        for b in BITS:
            for n in self.buckets: self.graphs[(b, n)] = self._capture(b, n)
        self.ctx = []                                                            # tokens whose K/V are in the cache
        eos = c.eos_token_id; self.eos = set(eos if isinstance(eos, list) else [eos]) | ({self.tok.eos_token_id} if self.tok.eos_token_id is not None else set())
        log(f"loaded {os.path.basename(os.path.normpath(ckpt))} in {time.time() - t0:.0f} s, VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GB")

    @staticmethod
    def _get(where, k):
        with safe_open(where[k], "pt") as f: return f.get_tensor(k)

    def _kernel_configs(self):
        """the fastest kernel and launch configuration per matrix shape and mode on this GPU (measured once, kept in demo/kernel_configs.json)"""
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_configs.json"); gpu = torch.cuda.get_device_name(0)
        allc = json.load(open(f)) if os.path.exists(f) else {}; mine = allc.setdefault(gpu, {}); x = {}
        for p in PROJ:
            m = self.lin[(0, p)]; key = f"{m.rows}x{m.cols}"
            if key in mine: continue
            self.log(f"tuning the kernels for {key} (once per GPU)"); ms = [self.lin[(l, p)] for l in range(self.NL)]
            xx = torch.randn(1, m.cols, device="cuda").half(); mine[key] = {}
            for b in LADDER:
                mine[key][str(b)] = min(CANDIDATES, key=lambda cand: timed([(lambda mm=mm, kw=parse(cand), k=K_OF[b]: mm(xx, k, **kw)) for mm in ms]))
            json.dump(allc, open(f, "w"), indent=1)
        return {p: {b: parse(mine[f"{self.lin[(0, p)].rows}x{self.lin[(0, p)].cols}"][str(min(b, 4))]) for b in BITS} for p in PROJ}   # 5-8 bits: the 4-bit kernel

    def _linear(self, l, p, x, b):
        m = self.lin[(l, p)]; x = x.contiguous(); y = m(x, K_OF[min(b, 4)], **self.kcfg[p][b])
        if b > 4: m.residual(x, y, int(b - 4))
        return y + self.bias[(l, p)] if (l, p) in self.bias else y

    def _rope(self, x, c, s):
        h = self.HD // 2; x1, x2 = x[..., :h], x[..., h:]; return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)

    def _step(self, b, n):
        NH, NKV, HD = self.NH, self.NKV, self.HD
        h = self.embed.index_select(0, self.tokbuf); c = self.cos.index_select(0, self.pos).view(1, 1, 1, -1); s = self.sin.index_select(0, self.pos).view(1, 1, 1, -1)
        m = self.mask[:, :n].index_select(0, self.pos).view(1, 1, 1, n)
        for l in range(self.NL):
            a = F.rms_norm(h, (self.D,), self.ln1[l], self.eps)
            q = self._rope(self._linear(l, PROJ[0], a, b).reshape(1, 1, NH, HD).transpose(1, 2), c, s)
            kk = self._rope(self._linear(l, PROJ[1], a, b).reshape(1, 1, NKV, HD).transpose(1, 2), c, s); v = self._linear(l, PROJ[2], a, b)
            self.kc[l].index_copy_(2, self.pos, kk); self.vc[l].index_copy_(2, self.pos, v.reshape(1, 1, NKV, HD).transpose(1, 2))
            o = F.scaled_dot_product_attention(q, self.kc[l][:, :, :n], self.vc[l][:, :, :n], attn_mask=m, enable_gqa=True)
            h = h + self._linear(l, PROJ[3], o.transpose(1, 2).reshape(1, NH * HD), b)
            a = F.rms_norm(h, (self.D,), self.ln2[l], self.eps)
            h = h + self._linear(l, PROJ[6], F.silu(self._linear(l, PROJ[4], a, b)) * self._linear(l, PROJ[5], a, b), b)
        self.logits.copy_(F.linear(F.rms_norm(h, (self.D,), self.lnf, self.eps), self.lm_head))

    @torch.no_grad()
    def _capture(self, b, n):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream()); self.pos.zero_(); self.tokbuf.zero_()
        with torch.cuda.stream(s):
            for _ in range(2): self._step(b, n)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=getattr(self, "pool", None)): self._step(b, n)
        self.pool = g.pool(); return g

    def close(self):
        """before the model is unloaded: the CUDA graphs keep their own memory pool until they are reset"""
        for g in self.graphs.values(): g.reset()
        self.graphs.clear(); torch.cuda.synchronize()

    def bytes_per_token(self, b):
        """bytes of the linear layers read per decoded token in mode b (planes + the table the kernel reads; 5-8 bits: + residual planes and offsets)"""
        k = K_OF[min(b, 4)]; r = int(b - 4) if b > 4 else 0; tot = 0
        for m in self.lin.values():
            tot += m.U.numel() + sum(m.H[p].numel() for p in range(k - 1)) + (m.codes.numel() + 4 * m.rows_pad if k == 7 else m.tab[k].numel() * 2)
            if r: tot += r * m.rows * m.cols // 8 + m.rows * (2 << r)
        return tot

    def stored_bytes(self):
        return sum(m.U.numel() + sum(h.numel() for h in m.H) + m.codes.numel() + 4 * m.rows_pad + (m.res[0].numel() + 2 * m.res[1].numel() if m.res else 0) for m in self.lin.values())

    def params(self): return sum(m.rows * m.cols for m in self.lin.values())

    @torch.no_grad()
    def _forward(self, token, b):
        p = len(self.ctx); assert p < self.lmax, "context full"
        self.tokbuf.fill_(token); self.pos.fill_(p); self.graphs[(b, next(n for n in self.buckets if n > p))].replay(); self.ctx.append(token)

    @torch.no_grad()
    def generate(self, ids, bits, max_new, on_token, greedy=False, temperature=0.7, top_p=0.9, stop=lambda: False):
        """continue the cached context (reused when `ids` extends it) with greedy or nucleus sampling; bits() is read before every token,
        so the mode can change during a reply. on_token(text piece, mode). Returns statistics."""
        n = 0
        while n < len(self.ctx) and n < len(ids) and self.ctx[n] == ids[n]: n += 1
        if n == len(ids): n -= 1                                                  # at least the last prompt token is processed again
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
