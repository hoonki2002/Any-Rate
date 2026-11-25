"""On-device decode speed of Qwen3-30B-A3B with a VRAM expert cache and experts read from the local NVMe on a miss.
Non-expert weights (attention, router, norms, embeddings, lm_head) stay in VRAM in bf16. Every layer keeps an LRU cache of S expert slots;
a missing expert is read with O_DIRECT (page cache bypassed, so host DRAM does not act as a second cache) into pinned memory and copied to its slot.
Expert formats: bf16 (the HF safetensors shards) or Any-Rate mode K (repacked files, see repack.py; grouped kernel kernels/csrc/ar_moe.cu).
usage: python -m moe.engine --fmt bf16 --model HF_DIR --cache_gb 6 [--tokens 64]
       python -m moe.engine --fmt ar --K 3 --ar_dir AR_DIR --model NONEXPERT_DIR --cache_gb 6      (python -m moe speed runs them all)"""
import argparse, collections, glob, json, mmap, os, struct, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, DynamicCache
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeRotaryEmbedding

PAGE = 4096
def ar_moe(planes, tabs, xs, ys, rows, cols, k):
    """grouped Any-Rate GEMV (kernels/lib/libar_moe.so, bash kernels/build.sh): planes: n lists of 7 data pointers (unused planes 0); tabs/xs/ys: n pointers"""
    import ctypes
    global _ARMOE
    if "_ARMOE" not in globals():
        _ARMOE = ctypes.CDLL(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernels", "lib", "libar_moe.so")); _ARMOE.ar_moe.restype = ctypes.c_int
    n = len(tabs); P = (ctypes.c_void_p * (7 * n))(*[p for pl in planes for p in pl]); T = (ctypes.c_void_p * n)(*tabs)
    X = (ctypes.c_void_p * n)(*xs); Y = (ctypes.c_void_p * n)(*ys); R = (ctypes.c_int * n)(*rows); C = (ctypes.c_int * n)(*cols)
    r = _ARMOE.ar_moe(P, T, X, Y, R, C, n, k, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)); assert r == 0, f"ar_moe error {r}"

class Energy:
    """GPU board energy from the NVML counter (mJ), CPU package energy from RAPL (uJ, needs read permission; skipped otherwise)"""
    def __init__(self):
        import ctypes; self.c = ctypes; self.nv = ctypes.CDLL("libnvidia-ml.so.1"); assert self.nv.nvmlInit_v2() == 0
        self.h = ctypes.c_void_p(); assert self.nv.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.h)) == 0
        self.rapl = "/sys/class/powercap/intel-rapl:0/energy_uj"
        try: open(self.rapl).read()
        except OSError: self.rapl = None
        self.rapl_range = int(open(self.rapl.replace("energy_uj", "max_energy_range_uj")).read()) / 1e6 if self.rapl else 0.0
    def read(self):   # (GPU J, CPU J or None)
        e = self.c.c_ulonglong(); assert self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h, self.c.byref(e)) == 0
        return e.value / 1e3, (int(open(self.rapl).read()) / 1e6 if self.rapl else None)
ENERGY = Energy()
def align_down(x): return x // PAGE * PAGE
def align_up(x): return (x + PAGE - 1) // PAGE * PAGE

class Staging:
    """one page-aligned pinned host buffer per worker thread"""
    def __init__(self, nbytes):
        self.raw = torch.empty(nbytes + PAGE, dtype=torch.uint8, pin_memory=True); off = (-self.raw.data_ptr()) % PAGE
        self.buf = self.raw[off:off + nbytes]; self.mv = memoryview(self.buf.numpy()); self.stream = torch.cuda.Stream()

class BF16Store:
    """experts of the HF safetensors shards; the three tensors of an expert are read with one O_DIRECT read of their aligned span"""
    def __init__(self, model_dir, cfg):
        self.I, self.H = cfg.moe_intermediate_size, cfg.hidden_size; self.wbytes = self.I * self.H * 2
        self.slot_bytes = 3 * self.wbytes; self.loc = {}; self.fds = {}
        for f in sorted(glob.glob(f"{model_dir}/model-*.safetensors")):
            with open(f, "rb") as fh: n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
            base = 8 + n; self.fds[f] = os.open(f, os.O_RDONLY | os.O_DIRECT)
            for k, v in hdr.items():
                if ".mlp.experts." not in k: continue
                p = k.split("."); li, e, proj = int(p[2]), int(p[5]), p[6]
                self.loc.setdefault((li, e), {})[proj] = (f, base + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0])
        self.ops = {}   # (layer, expert) -> [(file, aligned offset, aligned length, [(proj, offset inside the read, nbytes)])]: one read per shard when the tensors are adjacent
        for key, d in self.loc.items():
            ops = []
            for f in sorted({t[0] for t in d.values()}):
                ts = [(proj, o, n) for proj, (ff, o, n) in d.items() if ff == f]; lo = align_down(min(o for _, o, n in ts)); hi = align_up(max(o + n for _, o, n in ts))
                if hi - lo <= 1.05 * sum(align_up(n) + PAGE for _, o, n in ts): ops.append((f, lo, hi - lo, [(proj, o - lo, n) for proj, o, n in ts]))
                else: ops += [(f, align_down(o), align_up(o + n) - align_down(o), [(proj, o - align_down(o), n)]) for proj, o, n in ts]
            self.ops[key] = ops
        self.stage_bytes = max(sum(op[2] for op in ops) for ops in self.ops.values())
        print(f"bf16 store: {len(self.loc)} experts, reads per expert {collections.Counter(len(o) for o in self.ops.values())}, max staging {self.stage_bytes / 1e6:.2f} MB", flush=True)
    def read(self, li, e, st, slot):   # slot: uint8 GPU tensor [slot_bytes] -> gate | up | down
        pos, total, cp = 0, 0, []
        for f, off, ln, ts in self.ops[(li, e)]:
            got = os.preadv(self.fds[f], [st.mv[pos:pos + ln]], off); assert got == ln
            cp += [(proj, pos + o, n) for proj, o, n in ts]; pos += ln; total += ln
        with torch.cuda.stream(st.stream):
            for proj, o, n in cp:
                i = ("gate_proj", "up_proj", "down_proj").index(proj); slot[i * self.wbytes:(i + 1) * self.wbytes].copy_(st.buf[o:o + n], non_blocking=True)
        st.stream.synchronize(); return total
    def weights(self, slot):
        w = slot.view(torch.bfloat16); n = self.I * self.H
        return w[:n].view(self.I, self.H), w[n:2 * n].view(self.I, self.H), w[2 * n:].view(self.H, self.I)
    def compute(self, x, w, slots, pool):
        y = torch.zeros_like(x)
        for j, s in enumerate(slots):
            Wg, Wu, Wd = self.weights(pool[s]); y += w[j] * F.linear(F.silu(F.linear(x, Wg)) * F.linear(x, Wu), Wd)
        return y

class ARStore:
    """Any-Rate mode K (K planes: 3..7 -> 2, 2.5, 3, 3.5, 4 bits) of the repacked files (repack.py): per expert one O_DIRECT read of the plane prefix
    and one of the mode-K table. Slot layout = the two reads back to back; the experts are computed by the grouped kernel of ar_moe.cu."""
    def __init__(self, ar_dir, cfg, K):
        self.kern = ar_moe
        self.K, self.I, self.H = K, cfg.moe_intermediate_size, cfg.hidden_size; n = self.I * self.H
        self.usz, self.hsz = n // 8, n // 16; self.file_slot = align_up(3 * self.usz + 18 * self.hsz)
        self.plane_bytes = align_up(3 * self.usz + 3 * (K - 1) * self.hsz)
        tb = [r * ((1 << K) * 2 if K <= 6 else 132) for r in (self.I, self.I, self.H)]; self.tab_off = [self.plane_bytes + sum(tb[:i]) for i in range(3)]
        self.tab_slot = align_up(sum(tb)); self.slot_bytes = self.stage_bytes = self.plane_bytes + self.tab_slot
        self.fp = {li: os.open(f"{ar_dir}/L{li}.planes", os.O_RDONLY | os.O_DIRECT) for li in range(cfg.num_hidden_layers)}
        self.ft = {li: os.open(f"{ar_dir}/L{li}.tab{K}", os.O_RDONLY | os.O_DIRECT) for li in range(cfg.num_hidden_layers)}
        self.poff = [[(m * self.usz if p == 0 else 3 * self.usz + 3 * (p - 1) * self.hsz + m * self.hsz) if p < K else None for p in range(7)] for m in range(3)]
        print(f"AR store: mode K={K}, {self.slot_bytes / 1e6:.3f} MB per expert", flush=True)
    def read(self, li, e, st, slot):
        assert os.preadv(self.fp[li], [st.mv[:self.plane_bytes]], e * self.file_slot) == self.plane_bytes
        assert os.preadv(self.ft[li], [st.mv[self.plane_bytes:self.slot_bytes]], e * self.tab_slot) == self.tab_slot
        with torch.cuda.stream(st.stream): slot.copy_(st.buf[:self.slot_bytes], non_blocking=True)
        st.stream.synchronize(); return self.slot_bytes
    def _ptrs(self, b, m): return [b + o if o is not None else 0 for o in self.poff[m]], b + self.tab_off[m]
    def compute(self, x, w, slots, pool):
        I, H, k = self.I, self.H, len(slots); base = [pool[s].data_ptr() for s in slots]
        gu = torch.empty(2 * k, I, dtype=x.dtype, device=x.device); pl, tb = [], []
        for b in base:
            for m in (0, 1): p, t = self._ptrs(b, m); pl.append(p); tb.append(t)
        self.kern(pl, tb, [x.data_ptr()] * (2 * k), [gu[i].data_ptr() for i in range(2 * k)], [I] * (2 * k), [H] * (2 * k), self.K)
        act = (F.silu(gu[0::2]) * gu[1::2]).contiguous(); yd = torch.empty(k, H, dtype=x.dtype, device=x.device); pl, tb = [], []
        for b in base: p, t = self._ptrs(b, 2); pl.append(p); tb.append(t)
        self.kern(pl, tb, [act[j].data_ptr() for j in range(k)], [yd[j].data_ptr() for j in range(k)], [H] * k, [I] * k, self.K)
        return (w[:, None] * yd).sum(0, keepdim=True)

class ExpertCache:
    """per-layer LRU of S slots; misses of one layer are read in parallel by a thread pool"""
    def __init__(self, store, L, S, workers=8):
        self.store, self.S = store, S
        self.pool = [torch.empty(S, store.slot_bytes, dtype=torch.uint8, device="cuda") for _ in range(L)]
        self.lru = [collections.OrderedDict() for _ in range(L)]; self.free = [list(range(S)) for _ in range(L)]
        self.tls = threading.local(); self.ex = ThreadPoolExecutor(workers); self.stats = collections.Counter()
    def _stage(self):
        if not hasattr(self.tls, "st"): self.tls.st = Staging(self.store.stage_bytes)
        return self.tls.st
    def _load(self, li, e, s): return self.store.read(li, e, self._stage(), self.pool[li][s])
    def get(self, li, ids):
        lru = self.lru[li]; miss = []
        for e in ids:
            if e in lru: lru.move_to_end(e); self.stats["hit"] += 1
            else: miss.append(e)
        jobs = []
        for e in miss:
            if self.free[li]: s = self.free[li].pop()
            else: _, s = lru.popitem(last=False)
            lru[e] = s; jobs.append(self.ex.submit(self._load, li, e, s))
        t0 = time.perf_counter()
        for j in jobs: self.stats["bytes"] += j.result()
        self.stats["miss"] += len(miss); self.stats["io_s"] += time.perf_counter() - t0
        return [lru[e] for e in ids]

class CachedMoE(nn.Module):
    """Qwen3 sparse MoE block for one decode token: router in VRAM, the 8 selected experts from the cache"""
    def __init__(self, cfg, li, cache):
        super().__init__(); self.li, self.cache, self.k = li, cache, cfg.num_experts_per_tok; self.norm = cfg.norm_topk_prob
        self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
    def forward(self, h):
        b, t, d = h.shape; x = h.view(-1, d); assert x.shape[0] == 1, "decode only (one token per step)"
        p = F.softmax(self.gate(x), dim=-1, dtype=torch.float); w, ids = torch.topk(p, self.k, dim=-1)
        if self.norm: w = w / w.sum(-1, keepdim=True)
        w = w.to(x.dtype)[0]; ids = ids[0].tolist(); x = x.contiguous()
        if len(ids) > self.cache.S:   # cached experts first, so the misses fill whole groups and are read in parallel
            o = sorted(range(len(ids)), key=lambda j: ids[j] not in self.cache.lru[self.li]); ids = [ids[j] for j in o]; w = w[o]
        g = -(-len(ids) // self.cache.S); n = -(-len(ids) // g); y = None   # g groups of (nearly) equal size, each <= S
        for a in range(0, len(ids), n):   # fewer slots than selected experts: load and compute them in groups of n, the next group reusing the slots
            if a: torch.cuda.current_stream().synchronize()   # the previous group must finish reading its slots before they are overwritten
            part = self.cache.store.compute(x, w[a:a + n], self.cache.get(self.li, ids[a:a + n]), self.cache.pool[self.li]); y = part if y is None else y + part
        return y.view(b, t, d), None

def nonexpert_state(model_dir):
    f1 = f"{model_dir}/nonexpert.safetensors"; files = [f1] if os.path.exists(f1) else sorted(glob.glob(f"{model_dir}/model-*.safetensors")); sd = {}
    for f in files:
        with safe_open(f, "pt", device="cpu") as fh:   # read to host: a GPU copy next to the GPU parameters would double the load-time peak
            for k in fh.keys():
                if ".mlp.experts." not in k: sd[k] = fh.get_tensor(k)
    return sd

def build(model_dir, cache):
    cfg = AutoConfig.from_pretrained(model_dir); torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"): model = Qwen3MoeForCausalLM(cfg)
    for li, layer in enumerate(model.model.layers): layer.mlp = CachedMoE(cfg, li, cache)
    model.to_empty(device="cuda"); missing, unexpected = model.load_state_dict(nonexpert_state(model_dir), strict=False)
    assert not unexpected and not missing, (missing[:5], unexpected[:5])
    model.model.rotary_emb = Qwen3MoeRotaryEmbedding(cfg, device="cuda"); torch.set_default_dtype(torch.float32)
    return model.eval()

PROMPTS = ["Explain how a mixture-of-experts language model decides which experts to use for each token, and why this saves computation.",
           "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
           "List the main differences between TCP and UDP and give an example application for each."]

@torch.no_grad()
def run(model, tok, cache, prompt, ntok):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt").cuda(); kv = DynamicCache()
    for t in range(ids.shape[1]):   # prefill one token at a time (decode path), not timed
        out = model(input_ids=ids[:, t:t + 1], past_key_values=kv, use_cache=True, cache_position=torch.tensor([t], device="cuda"))
    nxt = out.logits[:, -1].argmax(-1, keepdim=True); pos = ids.shape[1]; gen = [nxt.item()]
    cache.stats.clear(); torch.cuda.synchronize(); g0, c0 = ENERGY.read(); t0 = time.perf_counter()
    for i in range(ntok - 1):
        out = model(input_ids=nxt, past_key_values=kv, use_cache=True, cache_position=torch.tensor([pos], device="cuda")); pos += 1
        nxt = out.logits[:, -1].argmax(-1, keepdim=True); gen.append(nxt.item())
    torch.cuda.synchronize(); dt = time.perf_counter() - t0; g1, c1 = ENERGY.read(); s = cache.stats; n = s["hit"] + s["miss"]; nt = ntok - 1
    r = dict(tps=nt / dt, hit=s["hit"] / max(1, n), io_frac=s["io_s"] / dt, gb_per_tok=s["bytes"] / 1e9 / nt, gpu_j_per_tok=(g1 - g0) / nt, gpu_w=(g1 - g0) / dt)
    if c0 is not None: e_ = c1 - c0 + (ENERGY.rapl_range if c1 < c0 else 0); r.update(cpu_j_per_tok=e_ / nt, cpu_w=e_ / dt)   # RAPL counter wraps at max_energy_range_uj
    return r, tok.decode(gen)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fmt", default="bf16"); ap.add_argument("--model", required=True); ap.add_argument("--cache_gb", type=float, required=True)
    ap.add_argument("--tokens", type=int, default=64); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--out")
    ap.add_argument("--ar_dir"); ap.add_argument("--K", type=int, default=3); a = ap.parse_args()
    cfg = AutoConfig.from_pretrained(a.model); L = cfg.num_hidden_layers
    store = BF16Store(a.model, cfg) if a.fmt == "bf16" else ARStore(a.ar_dir, cfg, a.K)
    S = min(cfg.num_experts, int(a.cache_gb * 1e9 / (L * store.slot_bytes))); assert S >= 1, "cache too small: no expert slot per layer"
    cache = ExpertCache(store, L, S, a.workers); model = build(a.model, cache); tok = AutoTokenizer.from_pretrained(a.model)
    print(f"fmt {a.fmt}  cache {a.cache_gb} GB = {S} experts/layer  VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    res = []
    for p in PROMPTS:
        r, text = run(model, tok, cache, p, a.tokens); res.append(r)
        print({k: round(v, 3) for k, v in r.items()}, "|", text[:80].replace("\n", " "), flush=True)
    summ = {k: float(np.mean([r[k] for r in res])) for k in res[0]}; summ.update(fmt=a.fmt if a.fmt == "bf16" else f"ar_K{a.K}", cache_gb=a.cache_gb, slots=S, tokens=a.tokens)
    print("MEAN", json.dumps(summ), flush=True)
    if a.out: json.dump(dict(summary=summ, runs=res), open(a.out, "w"), indent=1)
