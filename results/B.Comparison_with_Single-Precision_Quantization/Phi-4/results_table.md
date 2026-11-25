# Phi-4: single-precision GPTQ

PPL = WikiText-2. Benchmarks in %: ARC-C/ARC-E/HellaSwag/PIQA = acc_norm, WinoGrande = acc, GSM8K = flexible-extract exact match, TriviaQA/NQ-open = exact match. Avg8 = mean of the 8 benchmarks. One model per bit width; FP16, MG (2/3/4) and AR (ours): ../A.Full_Benchmark_Results.

| bit | method | PPL(WikiText-2) | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | GSM8K(flexible) | TriviaQA | NQ-open | Avg8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | GPTQ | 1648.33 | 24.32 | 31.94 | 31.70 | 55.77 | 51.14 | 2.05 | 0.00 | 0.10 | 24.63 |
| 3 | GPTQ | 7.19 | 54.27 | 70.83 | 79.62 | 80.79 | 71.27 | 84.61 | 59.80 | 23.20 | 65.55 |
| 4 | GPTQ | 6.59 | 55.38 | 71.42 | 81.71 | 81.01 | 74.43 | 89.08 | 67.30 | 28.70 | 68.63 |
