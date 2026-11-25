#!/usr/bin/env python3
"""Any-Rate chat demo: one checkpoint, and the bit width (2, 2.5, 3, 3.5, 4, 5, 6, 7 or 8 bits) is switched at any time, also in the middle
of a reply. The model is chosen on the page (one in VRAM at a time); --model sets the first one.

    bash kernels/build.sh                        # once
    python demo/moe_pack.py                      # once, optional: the experts of Qwen3-30B-A3B for the MoE model (about 40 GB on the NVMe)
    python demo/app.py                           # then open http://127.0.0.1:7860
    python demo/app.py --dry-run                 # the page only, no model (simulated numbers)

Models: Llama-3.1-8B-Instruct and Qwen2.5-3B (base) keep every linear layer in the Any-Rate format in VRAM, read by the Any-Rate GEMV
kernels (engine.py); Qwen3-30B-A3B-Instruct-2507 (MoE) keeps its experts on the NVMe and the ones in use in a VRAM cache (moe.py, --cache-gb,
also on the page). A checkpoint is downloaded from Hugging Face the first time it is chosen (--ckpt: a local folder for the first model). --host 0.0.0.0 opens the page to other devices on the network
(a phone or a laptop). Only the Python standard library is used for the web part; the page renders Markdown itself and the math with
KaTeX 0.16.11 (static/katex, MIT license), served by this server, so no internet connection is needed.

Power: the GPU from its energy counter (NVML), the CPU package from RAPL (/sys/class/powercap), sampled every 0.5 s; a sensor that
cannot be read is left out. The rest of the PC (DRAM, disk, fans, power supply) is not measured."""
import argparse, collections, ctypes, json, os, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEMO = os.path.dirname(os.path.abspath(__file__))
MODELS = {"llama-3.1-8b-instruct": {"label": "Llama-3.1-8B-Instruct", "repo": "HoonkiLee/Llama-3.1-8B-Instruct-AR2to8", "moe": False},
          "qwen3-30b-a3b": {"label": "Qwen3-30B-A3B-Instruct-2507 (MoE)", "repo": "HoonkiLee/Qwen3-30B-A3B-Instruct-2507-AR2to8", "moe": True},
          "qwen2.5-3b": {"label": "Qwen2.5-3B (base)", "repo": "HoonkiLee/Qwen2.5-3B-AR2to8", "moe": False}}
BITS = [2, 2.5, 3, 3.5, 4, 5, 6, 7, 8]
EXAMPLES = ["Explain in three sentences why the sky is blue.",
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. "
            "She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day? Think step by step.",
            "Write a short poem about a lighthouse."]
MAX_MESSAGES = 60
STATIC_TYPES = {".js": "text/javascript", ".css": "text/css", ".woff2": "font/woff2"}


def gpu_info():
    try:
        idx = (os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0").strip()
        name, total, used = [x.strip() for x in subprocess.run(["nvidia-smi", "-i", idx, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                                                               capture_output=True, text=True, timeout=5).stdout.strip().split(",")]
        return {"name": name, "total_gb": round(float(total) / 1024, 1), "used_gb": round(float(used) / 1024, 1)}
    except Exception:
        return None


class Energy:
    """cumulative energy in J of the GPU (NVML) and the CPU package (RAPL); None where the sensor cannot be read"""
    def __init__(self):
        self.nvml = self.h = None; self.rapl = None
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1"); h = ctypes.c_void_p(); idx = int((os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0").strip())
            if lib.nvmlInit_v2() == 0 and lib.nvmlDeviceGetHandleByIndex_v2(idx, ctypes.byref(h)) == 0: self.nvml, self.h = lib, h
            if self.gpu() is None: self.nvml = None
        except (OSError, ValueError): pass
        p = "/sys/class/powercap/intel-rapl:0"                    # package 0 (also on AMD CPUs)
        try:
            self.rapl_max = int(open(p + "/max_energy_range_uj").read()) / 1e6; self.rapl = p + "/energy_uj"; self.rapl_last, self.rapl_acc = self._read_rapl(), 0.0
        except (OSError, ValueError): self.rapl = None
        self.lock = threading.Lock()

    def gpu(self):
        e = ctypes.c_ulonglong()
        return e.value / 1e3 if self.nvml and self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.h, ctypes.byref(e)) == 0 else None

    def _read_rapl(self): return int(open(self.rapl).read()) / 1e6

    def cpu(self):                                                 # the RAPL counter wraps around at max_energy_range_uj
        if self.rapl is None: return None
        with self.lock:
            v = self._read_rapl(); d = v - self.rapl_last; self.rapl_last = v; self.rapl_acc += d if d >= 0 else d + self.rapl_max; return self.rapl_acc

    def read(self): return {"gpu": self.gpu(), "cpu": self.cpu()}


def joules(e0, e1):
    """energy between two readings: GPU, CPU and GPU + CPU (None if no sensor)"""
    d = {k: (e1[k] - e0[k]) if e0[k] is not None and e1[k] is not None else None for k in ("gpu", "cpu")}
    d["all"] = sum(v for v in d.values() if v is not None) if any(v is not None for v in d.values()) else None; return d


class Telemetry:
    """every 0.5 s: power of the GPU and the CPU, tokens decoded and their modes. A window that was decoding one reply from start to end in one
    mode counts towards that mode's averages (energy per token, tokens/s, W) until reset."""
    DT, KEEP = 0.5, 120

    def __init__(self):
        self.e = Energy(); self.samples = collections.deque(maxlen=self.KEEP); self.lock = threading.Lock(); self.counts = collections.Counter()
        self.decoding = False; self.reply = 0; self.reset(); threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _counters():                                              # the expert cache of the MoE: SSD bytes read, hits, misses
        f = getattr(State.backend, "counters", None); return collections.Counter(f()) if f else collections.Counter()

    def reset(self):
        with self.lock: self.modes = collections.defaultdict(lambda: {"s": 0.0, "tokens": 0, "gpu": 0.0, "cpu": 0.0, "all": 0.0, "ssd": 0})

    def token(self, b): self.counts[f"{b:g}"] += 1

    def _run(self):
        e0, t0, c0, d0, m0, r0 = self.e.read(), time.time(), collections.Counter(self.counts), self.decoding, self._counters(), self.reply
        while True:
            time.sleep(self.DT)
            e1, t1, c1, d1, m1, r1 = self.e.read(), time.time(), collections.Counter(self.counts), self.decoding, self._counters(), self.reply
            dt = t1 - t0; j = joules(e0, e1); n = {b: c1[b] - c0[b] for b in c1 if c1[b] > c0[b]}; tok = sum(n.values())
            s = {"t": t1, "dt": dt, "tokens": tok, "bits": next(iter(n)) if len(n) == 1 else ("mixed" if n else None), "full": d0 and d1 and r0 == r1,
                 **{f"{k}_w": (v / dt if v is not None else None) for k, v in j.items()}}
            if m1: s.update(ssd_mb=(m1["ssd_bytes"] - m0["ssd_bytes"]) / 1e6, hit=m1["hit"] - m0["hit"], miss=m1["miss"] - m0["miss"])
            with self.lock:
                self.samples.append(s)
                if s["full"] and len(n) == 1:
                    m = self.modes[s["bits"]]; m["s"] += dt; m["tokens"] += tok; m["ssd"] += s.get("ssd_mb", 0.0)
                    for k in ("gpu", "cpu", "all"): m[k] += j[k] or 0.0
            e0, t0, c0, d0, m0, r0 = e1, t1, c1, d1, m1, r1

    def snapshot(self):
        with self.lock:
            modes = {b: {"tokens": m["tokens"], "tokens_per_s": m["tokens"] / m["s"], **{f"{k}_mj": 1e3 * m[k] / m["tokens"] for k in ("gpu", "cpu", "all")},
                         **{f"{k}_w": m[k] / m["s"] for k in ("gpu", "cpu", "all")}, "ssd_mb": m["ssd"] / m["tokens"]} for b, m in self.modes.items() if m["tokens"]}
            return {"now": time.time(), "dt": self.DT, "samples": list(self.samples), "modes": modes,
                    "sensors": {"gpu": self.e.nvml is not None, "cpu": self.e.rapl is not None},
                    "moe": State.backend.moe_info() if hasattr(State.backend, "moe_info") else None}


class StopGeneration(Exception):
    pass


class RealBackend:
    def __init__(self, ckpt, lmax):
        sys.path.insert(0, os.path.dirname(DEMO))
        from demo.engine import Engine
        import torch
        self.torch = torch; self.e = Engine(ckpt, lmax=lmax); self.name = os.path.basename(os.path.normpath(ckpt))
        cfg = json.load(open(os.path.join(ckpt, "any_rate_config.json"))) if os.path.exists(os.path.join(ckpt, "any_rate_config.json")) else {}
        self.name = cfg.get("base_model", self.name); n = self.e.params()
        self.sizes = {"params": n, "fp16_gb": 2 * n / 1e9, "stored_gb": self.e.stored_bytes() / 1e9,
                      "read_gb": {f"{b:g}": self.e.bytes_per_token(b) / 1e9 for b in BITS}, "other_gb": self._other_bytes() / 1e9}

    def _other_bytes(self):
        e = self.e; return sum(t.numel() * 2 for t in [e.embed, e.lnf, *e.ln1, *e.ln2, *e.bias.values()]) + (0 if e.lm_head is e.embed else e.lm_head.numel() * 2)

    def prompt_ids(self, messages):
        tok = self.e.tok
        if tok.chat_template:
            return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        return tok("\n".join(m["content"] for m in messages), add_special_tokens=True).input_ids

    def generate(self, messages, bits, max_new, on_token, greedy, stop, fresh=False):
        if fresh: self.e.ctx = []
        ids = self.prompt_ids(messages)
        if len(ids) + max_new >= self.e.lmax: raise ValueError(f"the conversation is too long ({len(ids)} tokens; the context holds {self.e.lmax})")
        st = self.e.generate(ids, bits, max_new, on_token, greedy=greedy, stop=stop)
        st["read_gb_per_token"] = {f"{b:g}": round(self.sizes["read_gb"][f"{b:g}"], 2) for b in st["bits_used"]}; return st

    def vram_gb(self): return self.torch.cuda.memory_allocated() / 2**30

    def close(self): self.e.close(); self.e = None


class MoEBackend(RealBackend):
    """Qwen3-30B-A3B: experts on the NVMe (moe_pack.py), a VRAM expert cache of budget GB (moe.py)"""
    def __init__(self, d, lmax, cache_gb):
        sys.path.insert(0, os.path.dirname(DEMO))
        from demo.moe import MoEEngine
        import torch
        self.torch = torch; self.e = MoEEngine(d, cache_gb, lmax); f = os.path.join(d, "any_rate_config.json"); e = self.e
        self.name = json.load(open(f)).get("base_model") if os.path.exists(f) else "Qwen/Qwen3-30B-A3B-Instruct-2507"; n = e.params(); self.modes = e.modes
        self.sizes = {"params": n, "fp16_gb": 2 * n / 1e9, "stored_gb": e.stored_bytes() / 1e9, "read_gb": {f"{b:g}": e.bytes_per_token(b) / 1e9 for b in e.modes},
                      "other_gb": e.nonexpert_bytes / 1e9}
        free = torch.cuda.mem_get_info()[1] / 1e9 - torch.cuda.memory_allocated() / 1e9 - 1.5          # 1.5 GB: K/V cache and activations
        self.limits = {"min_gb": 1.25 * e.L * e.cfg.num_experts_per_tok * e.lay.expert_bytes(e.modes[-1]) / 1e9, "max_gb": max(cache_gb, int(free * 2) / 2)}

    def counters(self): return dict(self.e.cache.counters)

    def moe_info(self):
        c = self.e.cfg; return {**self.e.cache.info(), **self.limits, "experts_total": self.e.L * c.num_experts, "active": c.num_experts_per_tok,
                                "expert_mb": {f"{b:g}": self.e.lay.expert_bytes(b) / 1e6 for b in self.modes}}

    def resize(self, gb): self.e.cache.resize(min(max(gb, self.limits["min_gb"]), self.limits["max_gb"]) * 1e9)

    def close(self): self.e.cache.close(); self.e = None


class FakeBackend:
    name = "dry run (no model)"
    def __init__(self):
        n = 7.0e9; self.sizes = {"params": n, "fp16_gb": 14.0, "stored_gb": 3.84, "read_gb": {f"{b:g}": r for b, r in zip(BITS, (1.77, 2.23, 2.71, 3.23, 3.67, 4.55, 5.42, 6.29, 7.17))}, "other_gb": 2.1}
        self.words = "This is a simulated answer: no model is loaded in dry-run mode, so the text and the numbers only show how the page works.".split()
    def generate(self, messages, bits, max_new, on_token, greedy, stop, fresh=False):
        t0 = time.time(); used = set(); n = 0
        for i in range(min(max_new, 40)):
            if stop(): break
            b = bits(); used.add(b); time.sleep(1 / (150 - 10 * BITS.index(b))); on_token(self.words[i % len(self.words)] + " ", b); n += 1
        dt = time.time() - t0
        return {"prompt_tokens": 20, "context_tokens": 60, "kv_reused": False, "prefill_s": 0.1, "decode_s": dt, "tokens": n, "tokens_per_s": n / dt,
                "bits_used": sorted(used), "read_gb_per_token": {f"{b:g}": self.sizes["read_gb"][f"{b:g}"] for b in used}}
    def vram_gb(self): return 6.2
    def close(self): pass


class State:
    backend = None; lock = threading.Lock(); bits = 4.0; dry = False; stop = False; tel = None
    model = None; loading = None; error = None; args = None


def moe_dir(): sys.path.insert(0, os.path.dirname(DEMO)); from demo.moe_pack import DEFAULT_OUT; return DEFAULT_OUT


def model_list():
    out = []
    for k, m in MODELS.items():
        ok = not m["moe"] or os.path.exists(os.path.join(moe_dir(), "nonexpert.safetensors"))
        out.append({"key": k, "label": m["label"], "available": ok, "note": "" if ok else "run python demo/moe_pack.py first (about 40 GB on the NVMe)"})
    return out


def load(key, ckpt=None):
    """the backend of model key (downloaded from Hugging Face if needed)"""
    a = State.args
    if a.dry_run: return FakeBackend()
    if MODELS[key]["moe"]:
        ckpt = ckpt or moe_dir()
        if not os.path.exists(os.path.join(ckpt, "nonexpert.safetensors")): raise FileNotFoundError(f"{ckpt} has no repacked experts: run python demo/moe_pack.py first")
        return MoEBackend(ckpt, a.context, a.cache_gb)
    if ckpt is None:
        from huggingface_hub import snapshot_download
        print(f"downloading {MODELS[key]['repo']} ...", flush=True); ckpt = snapshot_download(MODELS[key]["repo"])
    return RealBackend(ckpt, a.context)


def switch(key):
    """unload the model in VRAM and load model key (in a thread; the page polls /api/info)"""
    try:
        old, State.backend = State.backend, None
        if old is not None:
            old.close(); del old
            import gc, torch; gc.collect(); torch.cuda.empty_cache()
        State.backend = load(key); State.model = key; State.bits = 4.0; State.tel.reset(); print(f"loaded {key}", flush=True)
    except Exception as exc:
        State.error = f"{type(exc).__name__}: {exc}"; print("loading failed:", State.error, flush=True)
    finally:
        State.loading = None; State.lock.release()


def modes(): return getattr(State.backend, "modes", BITS)       # the MoE without its residual files: 2-4 bits


class Handler(BaseHTTPRequestHandler):
    server_version = "AnyRateDemo/1.0"
    def log_message(self, *a): pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def _body(self):
        try: return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError: return {}

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):           # ?greedy and other page options are read by the page
            body = open(os.path.join(DEMO, "index.html"), "rb").read(); self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        elif self.path == "/api/info" and State.backend is None:      # a model is being loaded (or failed to load)
            self._json({"loading": State.loading, "error": State.error, "models": model_list(), "model_key": State.model})
        elif self.path == "/api/info":
            self._json({"model": State.backend.name, "model_key": State.model, "models": model_list(), "loading": None, "dry_run": State.dry, "bits": State.bits, "modes": modes(), "sizes": State.backend.sizes,
                        "examples": EXAMPLES, "gpu": gpu_info(), "vram_gb": round(State.backend.vram_gb(), 2),
                        "moe": State.backend.moe_info() if hasattr(State.backend, "moe_info") else None})
        elif self.path == "/api/hardware" and State.backend is not None:
            self._json({"gpu": gpu_info(), "vram_gb": round(State.backend.vram_gb(), 2), "sizes": State.backend.sizes})
        elif self.path == "/api/telemetry":
            self._json(State.tel.snapshot())
        elif self.path.startswith("/static/"):                       # KaTeX (static/katex), served locally so the page needs no internet
            root = os.path.join(DEMO, "static"); f = os.path.realpath(os.path.join(root, self.path[len("/static/"):].split("?")[0]))
            if not f.startswith(root + os.sep) or not os.path.isfile(f): return self.send_error(404)
            body = open(f, "rb").read(); self.send_response(200); self.send_header("Content-Type", STATIC_TYPES.get(os.path.splitext(f)[1], "application/octet-stream"))
            self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "max-age=86400"); self.end_headers(); self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/bits":                               # the slider: applied before the next token, also during a reply
            b = self._body().get("bits")
            if b not in modes(): return self._json({"error": f"bits must be one of {modes()}"}, 400)
            State.bits = float(b); return self._json({"bits": State.bits})
        if self.path == "/api/stop":
            State.stop = True; return self._json({"stopped": True})
        if self.path == "/api/cache":                              # the size of the VRAM expert cache (MoE)
            if State.backend is None or not hasattr(State.backend, "resize"): return self._json({"error": "not a MoE model"}, 400)
            gb = self._body().get("gb")
            if not isinstance(gb, (int, float)): return self._json({"error": "gb must be a number"}, 400)
            State.backend.resize(float(gb)); return self._json(State.backend.moe_info())
        if self.path == "/api/telemetry/reset":
            State.tel.reset(); return self._json({"reset": True})
        if self.path == "/api/model":                              # the model selector: unload, then load another one
            key = self._body().get("model")
            if key not in MODELS: return self._json({"error": f"model must be one of {list(MODELS)}"}, 400)
            if not next(m for m in model_list() if m["key"] == key)["available"]: return self._json({"error": next(m for m in model_list() if m["key"] == key)["note"]}, 400)
            if not State.lock.acquire(blocking=False): return self._json({"error": "a reply is being generated or a model is loading"}, 409)
            State.loading, State.error = key, None; threading.Thread(target=switch, args=(key,), daemon=True).start(); return self._json({"loading": key})
        if self.path != "/api/generate": return self.send_error(404)
        if State.backend is None: return self._json({"error": "no model is loaded"}, 503)
        req = self._body(); msgs = req.get("messages")
        ok = (isinstance(msgs, list) and 0 < len(msgs) <= MAX_MESSAGES and msgs[-1].get("role") == "user" and
              all(isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str) and 0 < len(m["content"]) <= 20000 for m in msgs))
        if not ok: return self._json({"error": "send a conversation that ends with a user message"}, 400)
        fixed = req.get("bits")                                     # compare tab: one fixed mode per request
        if fixed is not None and fixed not in modes(): return self._json({"error": f"bits must be one of {modes()}"}, 400)
        if not State.lock.acquire(blocking=False): return self._json({"error": "a reply is already being generated"}, 409)
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close"); self.end_headers()

        def send(obj):
            try: self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n"); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError): raise StopGeneration()
        State.stop = False; tel = State.tel; tel.reply += 1          # a window that spans two replies is not a full decoding window

        def on_token(t, b):
            tel.decoding = True; tel.token(b); send({"type": "token", "text": t, "bits": b})
        try:
            send({"type": "status", "text": "Generating ..."})
            bits = (lambda: float(fixed)) if fixed is not None else (lambda: State.bits)
            e0 = tel.e.read()
            st = State.backend.generate(msgs, bits, max(8, min(int(req.get("max_new_tokens", 512)), 2048)), on_token,
                                        bool(req.get("greedy", False)), lambda: State.stop, fresh=fixed is not None)
            st["energy_j"] = joules(e0, tel.e.read())                # the whole reply, prompt included
            send({"type": "done", "stats": st, "vram_gb": round(State.backend.vram_gb(), 2)})
        except StopGeneration:
            pass
        except Exception as exc:
            try: send({"type": "error", "text": f"{type(exc).__name__}: {exc}"})
            except StopGeneration: pass
        finally:
            tel.decoding = False; State.lock.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama-3.1-8b-instruct", choices=list(MODELS), help="the first model (others are chosen on the page)")
    ap.add_argument("--ckpt", help="local Any-Rate checkpoint folder of the first model (default: download it from Hugging Face)")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--context", type=int, default=4096, help="tokens the K/V cache holds (default %(default)s)")
    ap.add_argument("--dry-run", action="store_true", help="the page only: no model, simulated numbers")
    ap.add_argument("--cache-gb", type=float, default=4.0, help="qwen3-30b-a3b: VRAM expert cache in GB (default %(default)s; also on the page)")
    a = ap.parse_args(); State.dry, State.args = a.dry_run, a
    try: State.backend = load(a.model, a.ckpt)
    except FileNotFoundError as exc: sys.exit(str(exc))
    State.model = a.model; State.tel = Telemetry()
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"\nAny-Rate demo is running: http://{a.host}:{a.port}   (Ctrl-C to stop)", flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nbye")


if __name__ == "__main__":
    main()
