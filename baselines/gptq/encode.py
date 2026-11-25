"""Single-precision GPTQ (Frantar et al., ICLR 2023) with the authors' code, one model per bit width: the GPTQ class and Quantizer of
IST-DASLab/gptq on the Hessians of the Any-Rate calibration of the model (C4, 128 x 2048 tokens; any_rate/models/decoder.py), so that
GPTQ, MG and Any-Rate see the same Hessians. Asymmetric uniform grid, group 128, act-order, damping 0.01, block 128.

  python -m baselines.gptq.encode --model qwen2.5-3b

One calibration pass quantizes every linear layer at 2, 3 and 4 bits. Output (<original name>-GPTQ): <bits>bit/<module>/data.pt with
the dense fp16 weight of that bit width; dense.py reads one of them."""
import argparse, json, os, sys
import torch
from .. import common

BITS = (2, 3, 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=common.LLMS)
    ap.add_argument("--base", help="local folder of the original checkpoint (default: download it; Gemma-3-4b-it: the multimodal checkpoint)")
    ap.add_argument("--out", help="output folder (default: <original name>-GPTQ)")
    ap.add_argument("--layers", type=int, nargs="*", help="only these decoder layers (a quick check)")
    a = ap.parse_args()
    out = common.out_dir(a.model, "GPTQ", a.out); os.makedirs(out, exist_ok=True)
    json.dump({"method": "gptq", "bits": list(BITS), "group_size": 128, "sym": False, "act_order": True, "damping": 0.01}, open(os.path.join(out, "baseline_config.json"), "w"))
    sys.path.insert(0, common.code("gptq"))
    from gptq import GPTQ
    from quant import Quantizer
    from any_rate.encode import run_encoder
    if a.base: root = a.base
    else:                                                                 # the decoder encoder takes the original folder (Gemma-3: multimodal)
        from huggingface_hub import snapshot_download
        from any_rate.encode import HF_IDS
        common.log(f"downloading {HF_IDS[a.model]}"); root = snapshot_download(HF_IDS[a.model])

    def gptq_layer(W, H):
        q = {}
        for b in BITS:
            lin = torch.nn.Linear(W.shape[1], W.shape[0], bias=False, device=W.device, dtype=torch.float32); lin.weight.data = W.clone()
            g = GPTQ(lin); g.H = H.float().clone()
            g.quantizer = Quantizer(); g.quantizer.configure(b, perchannel=True, sym=False, mse=False)
            g.fasterquant(blocksize=128, percdamp=0.01, groupsize=128, actorder=True)
            q[b] = lin.weight.data.half().cpu(); del g, lin
        return q

    def save(i, store):
        for name, q in store.items():
            for b, w in q.items():
                d = os.path.join(out, f"{b}bit", name); os.makedirs(d, exist_ok=True); torch.save({"weight": w}, os.path.join(d, "data.pt"))
        open(os.path.join(out, f"decoder_{i}.done"), "w").close()

    run_encoder("decoder", root, "", out, a.layers, common.log, encode_layer=gptq_layer, save=save)
    common.log(f"done: {out}")


if __name__ == "__main__":
    main()
