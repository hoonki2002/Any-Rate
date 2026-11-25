#!/bin/bash
# Clone the authors' code of the baselines at the commits used in the paper and apply our patches (model support only; see README.md).
#   bash baselines/setup.sh            all four
#   bash baselines/setup.sh ap mg      some of them (ap: Any-Precision LLM, ab: AnyBCQ, mg: MatGPTQ, gptq: GPTQ, used unpatched)
#   bash baselines/setup.sh kernels    also build the authors' GEMV kernels (for the kernel and throughput benchmarks)
# Any-Precision LLM needs older numpy / datasets, so it gets its own environment (baselines/third_party/venv_ap).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; T="$HERE/third_party"; mkdir -p "$T"; PY=$(command -v python || command -v python3)
clone() {   # folder url commit patch
  if [ -d "$T/$1" ]; then echo "$T/$1 exists, skipped"; return; fi
  git clone -q "$2" "$T/$1" && git -C "$T/$1" checkout -q "$3"
  if [ -n "$4" ]; then git -C "$T/$1" apply "$4"; echo "$T/$1: $2 at $3, patched with $(basename "$4")"; else echo "$T/$1: $2 at $3"; fi
}
AP=(any-precision-llm https://github.com/SNU-ARC/any-precision-llm.git ee52889 "$HERE/any_precision/any_precision.patch")
AB=(anybcq https://github.com/naver-aics/anybcq.git c316a4d "$HERE/anybcq/anybcq.patch")
MG=(MatGPTQ https://github.com/IST-DASLab/MatGPTQ.git 86ee3ee "$HERE/matgptq/matgptq.patch")
GQ=(gptq https://github.com/IST-DASLab/gptq.git 2d65066 "")
for m in ${@:-ap ab mg gptq}; do
  case $m in
    ap) clone "${AP[@]}"
        if [ ! -x "$T/venv_ap/bin/python" ]; then
          $PY -m venv "$T/venv_ap" && "$T/venv_ap/bin/python" -m pip install -q -r "$HERE/any_precision/requirements.txt"
        fi ;;
    ab) clone "${AB[@]}" ;;
    mg) clone "${MG[@]}" ;;
    gptq) clone "${GQ[@]}" ;;
    kernels)   # the authors' GEMV kernels, unmodified, for kernels/bench.py and kernels/tps.py (built in the current environment)
        clone "${AP[@]}"; clone "${AB[@]}"; clone "${MG[@]}"
        (cd "$T/any-precision-llm/any_precision/modules/kernels" && $PY setup.py build_ext --inplace)
        CC_=$($PY -c "import torch; a, b = torch.cuda.get_device_capability(0); print(f'{a}{b}')")   # AnyBCQ lists sm_80 / sm_90 only: add this GPU
        (cd "$T/anybcq/anybcq/inference/custom_kernel" && NVCC_APPEND_FLAGS="-gencode=arch=compute_$CC_,code=sm_$CC_" MAX_JOBS=8 $PY setup.py build_ext --inplace)
        (cd "$T/MatGPTQ" && MAX_JOBS=8 $PY setup.py build_ext --inplace) ;;   # in place: imported from the source folders
    *) echo "unknown baseline $m (ap, ab, mg, gptq, kernels)"; exit 1 ;;
  esac
done
