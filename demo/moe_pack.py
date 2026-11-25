"""Repack the Any-Rate checkpoint of Qwen3-30B-A3B-Instruct-2507 for the demo, whose experts are read from the NVMe with O_DIRECT (moe.py).

    python demo/moe_pack.py [--out DIR] [--keep]

Per layer i, one page-aligned slot per expert in every file:
  L<i>.planes   U of gate, up, down, then H1 .. H6 of each (as moe/__main__.py repack): mode K (2-4 bits) reads the slot prefix
  L<i>.tab<K>   K = 3..7, the table of mode K for gate, up, down (fp16 [rows, 2^K], the mean of the children of the 8-bit table; K = 7: the
                8-bit codes, lo and step)
  L<i>.res      residual planes 1..4 of gate, up, down, plane by plane: 4 + r bits reads the slot prefix of r planes
  L<i>.lev<r>   r = 1..4, the residual offsets of the r-bit prefixes (fp16 [rows, 2^r]) for gate, up, down
and nonexpert.safetensors (attention, router, norms, embeddings, lm_head) with the config and tokenizer files. The shards are downloaded one
by one and deleted once their layers are written (--keep keeps them), so about 40 GB of disk is enough; an interrupted run continues."""
import argparse, json, os, re, shutil, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download

REPO = "HoonkiLee/Qwen3-30B-A3B-Instruct-2507-AR2to8"
DEFAULT_OUT = os.path.expanduser("~/.cache/any_rate/qwen3-30b-a3b-instruct-2507")
E, NL, PROJ, PAGE = 128, 48, ("gate_proj", "up_proj", "down_proj"), 4096
OFF = {1: 0, 2: 2, 3: 6, 4: 14}                     # res_lev columns of the r-bit prefixes
pad = lambda n: bytes((-n) % PAGE)
log = lambda *x: print(time.strftime("[%H:%M:%S]"), *x, flush=True)


def tab(P, K):
    codes, lo, st = P["table_codes"], P["table_lo"].float(), P["table_step"].float()
    if K == 7: return torch.cat([codes.contiguous().view(-1), lo.half().view(-1).view(torch.uint8), st.half().view(-1).view(torch.uint8)])
    raw = lo + codes.float() * st; g = 1 << (7 - K); return (raw.view(raw.shape[0], 1 << K, g).sum(-1) / g).half().contiguous().view(-1).view(torch.uint8)


def write(path, slots):
    with open(path + ".tmp", "wb") as f:
        for b in slots: f.write(b.numpy().tobytes() + pad(b.numel()))
    os.replace(path + ".tmp", path)


def pack_layer(get, li, out):
    P = [{pr: get(li, e, pr) for pr in PROJ} for e in range(E)]
    write(f"{out}/L{li}.planes", [torch.cat([P[e][pr][f"plane{p}"].contiguous().view(-1) for p in range(7) for pr in PROJ]) for e in range(E)])
    for K in range(3, 8): write(f"{out}/L{li}.tab{K}", [torch.cat([tab(P[e][pr], K) for pr in PROJ]) for e in range(E)])
    write(f"{out}/L{li}.res", [torch.cat([P[e][pr]["res_planes"][r].contiguous().view(-1) for r in range(4) for pr in PROJ]) for e in range(E)])
    for r in range(1, 5):
        write(f"{out}/L{li}.lev{r}", [torch.cat([P[e][pr]["res_lev"][:, OFF[r]:OFF[r] + (1 << r)].contiguous().view(-1).view(torch.uint8) for pr in PROJ]) for e in range(E)])
    open(f"{out}/L{li}.done", "w").close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT); ap.add_argument("--keep", action="store_true", help="keep the downloaded shards")
    a = ap.parse_args(); out = a.out; tmp = os.path.join(out, "_shards"); os.makedirs(tmp, exist_ok=True)
    for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "LICENSE", "any_rate_config.json"):
        if not os.path.exists(os.path.join(out, f)): shutil.copy(hf_hub_download(REPO, f), os.path.join(out, f))
    wmap = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))["weight_map"]
    layer = lambda k: int(re.search(r"layers\.(\d+)\.", k).group(1)) if "layers." in k else -1
    shards = sorted(set(wmap.values())); last = {s: max(layer(k) for k, f in wmap.items() if f == s) for s in shards}
    need = {li: {f for k, f in wmap.items() if layer(k) == li} for li in range(NL)}                # the shards that hold layer li
    ne_parts = os.path.join(out, "_nonexpert"); os.makedirs(ne_parts, exist_ok=True); handles = {}

    def get(li, e, pr):
        t = {}
        for field in ("plane0", "plane1", "plane2", "plane3", "plane4", "plane5", "plane6", "table_codes", "table_lo", "table_step", "res_planes", "res_lev"):
            k = f"model.layers.{li}.mlp.experts.{e}.{pr}.ar.{field}"; t[field] = handles[wmap[k]].get_tensor(k)
        return t

    for s in shards:
        if all(os.path.exists(f"{out}/L{li}.done") for li in range(NL) if li <= last[s]) and os.path.exists(os.path.join(ne_parts, s)): continue
        local = os.path.join(tmp, s)
        if not os.path.exists(local): log("downloading", s); hf_hub_download(REPO, s, local_dir=tmp)
        handles[s] = safe_open(local, "pt")
        if not os.path.exists(os.path.join(ne_parts, s)):
            save_file({k: handles[s].get_tensor(k) for k in handles[s].keys() if ".mlp.experts." not in k}, os.path.join(ne_parts, s) + ".tmp", metadata={"format": "pt"})
            os.replace(os.path.join(ne_parts, s) + ".tmp", os.path.join(ne_parts, s))
        for li in range(NL):                     # every layer whose tensors are all in the shards read so far
            if os.path.exists(f"{out}/L{li}.done") or not need[li] <= set(handles): continue
            pack_layer(get, li, out); log(f"layer {li} written")
        for d in [x for x in handles if all(os.path.exists(f"{out}/L{li}.done") for li in range(NL) if li <= last[x])]:
            del handles[d]
            if not a.keep: os.remove(os.path.join(tmp, d)); log("deleted", d)
    sd = {}
    for s in shards:
        with safe_open(os.path.join(ne_parts, s), "pt") as fh: sd.update({k: fh.get_tensor(k) for k in fh.keys()})
    save_file(sd, os.path.join(out, "nonexpert.safetensors"), metadata={"format": "pt"}); shutil.rmtree(ne_parts)
    if not a.keep: shutil.rmtree(tmp, ignore_errors=True)
    log("done:", out)


if __name__ == "__main__":
    main()
