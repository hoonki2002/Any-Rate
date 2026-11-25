"""MatGPTQ (MG; Kleinegger et al., 2026) with the authors' code: nested integer slices of one master code (GPTQ with act-order and
scale search, group 128). Two recipes, both calibrated on C4, 128 x 2048 tokens, seed 0:
  --recipe 348   the authors' official recipe: 8-bit master, slices 3/4/8 with weights 1/2/4
  --recipe 234   MG (2/3/4): 4-bit master, slices 2/3/4 with weights 1/1/1

  python -m baselines.matgptq.encode --model qwen2.5-3b --recipe 234

Language models are quantized by the authors' quant.py. Qwen3-30B-A3B: only the routed experts, bf16, with the expert rule of the
Any-Rate MoE encoding (identity Hessian for an expert that receives no calibration token, an isotropic prior of 512 tokens for one
that receives fewer). Whisper, PixArt-Sigma and CogVideoX-2B: the authors' MatGPTQ class on the Hessians of the Any-Rate calibration
(any_rate/models), with the same damping fallback. Output (<original name>-MG234 / -MG348): <module>/data.pt per layer
(qweight, scale, zero, perm); dense.py reads a mode."""
import argparse, json, os, sys
import torch
from .. import common
from any_rate.encode import MODELS

RECIPES = {"348": (8, ["3", "4", "8"], ["1", "2", "4"]), "234": (4, ["2", "3", "4"], ["1", "1", "1"])}
HOOKED = ["whisper-large-v3", "pixart-sigma", "cogvideox-2b"]
LINEARS = {"phi-4": ".*layers.*((qkv|o|gate_up|down)_proj)$", "qwen3-30b-a3b-instruct-2507": r".*layers.*mlp\.experts\.\d+\.(gate|up|down)_proj$"}


def quant_py(model, src, out, recipe, weights=None):
    master, opts, wts = RECIPES[recipe]; wts = weights or wts
    pre = ["model.embed_tokens"] + ([] if model in ("phi-4", "gemma-3-4b-it") else ["model.rotary_emb"])
    moe = model == "qwen3-30b-a3b-instruct-2507"; bf16 = moe or model == "gemma-3-4b-it"
    args = ["--model_name_or_path", src, "--quantizable_modules", LINEARS.get(model, ".*layers.*((q|k|v|o|gate|up|down)_proj)$"),
            "--pre_block_modules", *pre, "--block_modules", "model.layers", "--post_block_modules", "model.norm", "lm_head",
            "--calibration_data", "c4", "--calibration_tokens", "262144", "--calibration_sequence_length", "2048",
            "--method", "matgptq", "--master_bitwidth", str(master), "--bitwidth_options", *opts, "--bitwidth_weights", *wts,
            "--group_size", "128", "--perchannel", "--act_order", "--scale_search", "--seed", "0", "--verbose",
            "--dtype", "bfloat16" if bf16 else "float16", "--attn_implementation", "eager" if model == "gemma-3-4b-it" else "sdpa", "--save_dir", out]
    if moe: args.append("--low_cpu_mem_usage")
    common.run([sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc-per-node=1", "--master_port", "29571",
                os.path.join(common.HERE, "launch.py"), "--stand-in", "wandb", "quant.py"] + args, cwd=common.code("MatGPTQ"),
               env={"MOE_PRIOR": "512" if moe else "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, logfile=os.path.join(out, "quant.log"))


def hooked(model, root, out):
    """MG (2/3/4) of every linear layer that Any-Rate encodes, on the Hessians of the Any-Rate calibration of this model"""
    sys.path.insert(0, common.code("MatGPTQ")); os.environ["MOE_PRIOR"] = "0"
    from src.mat_gptq import MatGPTQ
    from any_rate.encode import run_encoder

    def mg_layer(W, H):
        lin = torch.nn.Linear(W.shape[1], W.shape[0], bias=False, device=W.device, dtype=torch.float16); lin.weight.data.copy_(W)
        for damp in (1e-2, 1e-1, 1.0):           # the official damping first; a larger one only if the Cholesky of the inverse Hessian fails
            try:
                mg = MatGPTQ(lin, [2, 3, 4], {2: 1, 3: 1, 4: 1}, perchannel=True, group_size=128, act_order=True, scale_search=True, rel_damp=damp)
                mg.H = H.clone(); mg.num_samples = 1; mg.quantization_pre_step(); qw, sc, ze, perm = mg.step()   # the scale of H does not matter
                if damp != 1e-2: common.log(f"   damping {damp} used for a {tuple(W.shape)} layer")
                return {"qweight": qw.cpu(), "scale": sc.cpu(), "zero": ze.cpu(), "perm": perm.cpu()}
            except torch._C._LinAlgError:
                pass
        raise RuntimeError("Cholesky failed for every damping value")

    for sub, kind, _ in MODELS[model]:
        work = os.path.join(out, sub or "model"); os.makedirs(work, exist_ok=True)
        def save(i, store, work=work, kind=kind):
            for name, t in store.items():
                os.makedirs(os.path.join(work, name), exist_ok=True); torch.save(t, os.path.join(work, name, "data.pt"))
            open(os.path.join(work, f"{kind}_{i}.done"), "w").close()
        common.log(f"MG (2/3/4) of {kind}"); run_encoder(kind, root, sub, work, None, common.log, encode_layer=mg_layer, save=save)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=common.LLMS + ["qwen3-30b-a3b-instruct-2507"] + HOOKED)
    ap.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    ap.add_argument("--base", help="local folder of the original checkpoint (default: download it)")
    ap.add_argument("--weights", nargs=3, help="other slice weights (the analysis also uses MG (2/3/4) with 1 2 4); output <original name>-MG<recipe>-w<weights>")
    ap.add_argument("--out", help="output folder (default: <original name>-MG<recipe>)")
    a = ap.parse_args()
    if a.recipe == "348" and a.model not in common.LLMS: sys.exit("the paper uses MG (2/3/4) only for this model: --recipe 234")
    if a.weights and a.model not in common.LLMS: sys.exit("--weights: language models only")
    out = common.out_dir(a.model, f"MG{a.recipe}" + (f"-w{''.join(a.weights)}" if a.weights else ""), a.out); os.makedirs(out, exist_ok=True)
    json.dump({"method": "matgptq", "recipe": a.recipe, "master_bitwidth": RECIPES[a.recipe][0]}, open(os.path.join(out, "baseline_config.json"), "w"))
    if a.model in HOOKED:
        hooked(a.model, a.base or common.original(a.model), out)
    else:
        quant_py(a.model, common.original(a.model, a.base), out, a.recipe, a.weights)
    common.log("written", out)


if __name__ == "__main__":
    main()
