#!/bin/bash
# Build the Any-Rate decode GEMV kernels (csrc/*.cu) into kernels/lib/. ARCH: the GPU architecture (default: that of GPU 0, e.g. sm_120).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; mkdir -p "$HERE/lib"
PY=$(command -v python || command -v python3)
ARCH=${ARCH:-sm_$($PY -c "import torch; a, b = torch.cuda.get_device_capability(0); print(f'{a}{b}')")}
for f in "$HERE"/csrc/*.cu; do
  n=$(basename "$f" .cu); nvcc -O3 -arch="$ARCH" -shared -Xcompiler -fPIC -o "$HERE/lib/lib$n.so" "$f" && echo "built lib/lib$n.so ($ARCH)"
done
