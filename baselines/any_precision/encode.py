"""Any-Precision LLM (AP; Park et al., ICML 2024) with the authors' code: an 8-bit parent upscaled from a 2-bit seed by incremental
upscaling, read at 2..8 bits. The authors' defaults except the seed precision (2 instead of 3): gradients on C4, 100 x 512 tokens.

  python -m baselines.any_precision.encode --model qwen2.5-3b

Output (<original name>-AP): the authors' quantize-mode cache, quantized/<run>/weights (8-bit codes) and lut_<b> (tables of each
mode); dense.py reads a mode."""
import argparse, os
from .. import common


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=common.LLMS)
    ap.add_argument("--base", help="local folder of the original checkpoint (default: download it)")
    ap.add_argument("--out", help="output folder (default: <original name>-AP)")
    a = ap.parse_args()
    src = common.original(a.model, a.base); out = common.out_dir(a.model, "AP", a.out); os.makedirs(out, exist_ok=True)
    root = common.code("any-precision-llm"); py = os.path.join(common.THIRD, "venv_ap", "bin", "python")
    common.run([py, "quantize.py", src, "--seed_precision", "2", "--parent_precision", "8", "--mode", "quantize", "--cache_dir", out,
                "--dataset", "c4", "--seq_len", "512", "--num_examples", "100"], cwd=root, logfile=os.path.join(out, "quantize.log"))
    common.log("written", out)


if __name__ == "__main__":
    main()
