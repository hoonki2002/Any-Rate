"""AnyBCQ (AB; Park et al., ICLR 2026) with the authors' code: binary-coded planes with per-precision scales, 2 -> 4 bits, trained by
block-wise reconstruction. The PTQ recipe of the authors' README: 512 C4 samples, 5000 iterations, group 128, symmetric, no beta
training, input_prob 0.5 (Llama, Gemma) or 1.0 (Qwen, Phi). TF32 matmuls; Gemma-3 in bf16 (fp16 overflows).

  python -m baselines.anybcq.encode --model qwen2.5-3b

Output (<original name>-AB2to4): the authors' checkpoint (qweight, alpha_<b>, beta_<b> per layer); dense.py reads a mode."""
import argparse, os, sys
from .. import common

INPUT_PROB = {"qwen2.5-3b": "1.0", "gemma-3-4b-it": "0.5", "llama-3.1-8b-instruct": "0.5", "phi-4": "1.0"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=common.LLMS)
    ap.add_argument("--base", help="local folder of the original checkpoint (default: download it)")
    ap.add_argument("--out", help="output folder (default: <original name>-AB2to4)")
    a = ap.parse_args()
    src = common.original(a.model, a.base); out = common.out_dir(a.model, "AB2to4", a.out); os.makedirs(out, exist_ok=True)
    args = ["--model_name_or_path", src, "--use_fast_tokenizer", "True", "--dataset_name", "c4", "--per_device_train_batch_size", "2",
            "--per_device_eval_batch_size", "8", "--w_lr1", "1e-4", "--w_lr2", "1e-4", "--w_lr3", "1e-4", "--n_bits_w", "2", "--add_bits", "2",
            "--group_size", "128", "--num_samples", "512", "--iters_w", "5000", "--input_prob", INPUT_PROB[a.model], "--asymmetric", "False",
            "--train_beta", "False", "--output_dir", out, "--cache_dir", os.path.join(out, "_cache")]
    if a.model == "gemma-3-4b-it": args += ["--torch_dtype", "bfloat16"]
    common.run([sys.executable, os.path.join(common.HERE, "launch.py"), "--tf32", "--stand-in", "custom_kernel", "run_clm.py"] + args, cwd=common.code("anybcq"),
               env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, logfile=os.path.join(out, "train.log"))
    common.log("written", out)


if __name__ == "__main__":
    main()
