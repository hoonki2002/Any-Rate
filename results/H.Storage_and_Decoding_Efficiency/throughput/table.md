# Decoding throughput (tokens/s), Llama-3.1-8B-Instruct, batch 1, RTX 5070 Ti

| method | 2 | 2.5 | 3 | 3.5 | 4 |
|---|---|---|---|---|---|
| AP | 141.63 | - | 124.07 | - | 107.57 |
| AB | 135.47 | - | 116.04 | - | 100.92 |
| MG | 132.51 | - | 119.32 | - | 106.92 |
| MG+perm | 123.70 | - | 112.06 | - | 101.02 |
| AR | 142.35 | 126.98 | 121.32 | 112.70 | 104.91 |

Without linear layers: 233.56 tokens/s (4.3 ms per token).
