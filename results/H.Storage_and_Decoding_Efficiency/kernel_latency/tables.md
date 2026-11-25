# GEMV time per linear layer (us), Llama-3.1-8B-Instruct, batch 1, RTX 5070 Ti

| layer (rows x columns) | method | 2 | 2.5 | 3 | 3.5 | 4 |
|---|---|---|---|---|---|---|
| q/o (4096x4096) | FP16 (cuBLAS) | 41.4 | | | | |
| | AP | 7.9 | - | 10.3 | - | 13.7 |
| | AB | 9.5 | - | 12.1 | - | 15.0 |
| | MG | 8.7 | - | 11.0 | - | 12.5 |
| | MG+perm | 12.3 | - | 14.6 | - | 15.8 |
| | AR | 8.5 | 11.4 | 12.2 | 13.1 | 13.8 |
| k/v (1024x4096) | FP16 (cuBLAS) | 12.5 | | | | |
| | AP | 3.8 | - | 4.3 | - | 5.0 |
| | AB | 6.1 | - | 7.0 | - | 8.0 |
| | MG | 3.4 | - | 4.0 | - | 4.7 |
| | MG+perm | 7.2 | - | 7.8 | - | 8.4 |
| | AR | 4.9 | 5.4 | 6.0 | 6.6 | 6.6 |
| gate/up (14336x4096) | FP16 (cuBLAS) | 140.5 | | | | |
| | AP | 21.0 | - | 29.5 | - | 38.6 |
| | AB | 25.8 | - | 35.7 | - | 46.3 |
| | MG | 26.9 | - | 35.0 | - | 43.4 |
| | MG+perm | 30.7 | - | 38.8 | - | 47.2 |
| | AR | 23.0 | 29.9 | 33.6 | 38.0 | 42.5 |
| down (4096x14336) | FP16 (cuBLAS) | 143.6 | | | | |
| | AP | 23.3 | - | 31.9 | - | 46.0 |
| | AB | 25.7 | - | 35.6 | - | 45.9 |
| | MG | 24.4 | - | 31.8 | - | 40.3 |
| | MG+perm | 28.3 | - | 35.9 | - | 44.3 |
| | AR | 23.6 | 30.4 | 33.2 | 37.7 | 43.4 |

# Linear-layer GEMV time per decoded token (ms)

| method | 2 | 2.5 | 3 | 3.5 | 4 |
|---|---|---|---|---|---|
| AP | 2.84 | - | 3.85 | - | 5.14 |
| AB | 3.46 | - | 4.64 | - | 5.90 |
| MG | 3.27 | - | 4.22 | - | 5.16 |
| MG+perm | 4.12 | - | 5.06 | - | 5.99 |
| AR | 3.08 | 3.96 | 4.37 | 4.90 | 5.41 |

MG+perm: MG with the act_order input permutation of its released execution path.
AP kernel of the released AB code vs the released AP kernel, profiler time at 3 and 4 bits: max difference 0.42%
