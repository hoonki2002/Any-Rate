# Qwen2.5-3B-base: single-precision GPTQ

PPL = WikiText-2. Benchmarks in %: ARC-C/ARC-E/HellaSwag/PIQA = acc_norm, WinoGrande = acc, GSM8K = flexible-extract exact match, TriviaQA/NQ-open = exact match. Avg8 = mean of the 8 benchmarks. One model per bit width; FP16, MG (2/3/4) and AR (ours): ../A.Full_Benchmark_Results.

| bit | method | PPL(WikiText-2) | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | GSM8K(flexible) | TriviaQA | NQ-open | Avg8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | GPTQ | 178379.74 | 27.56 | 25.76 | 26.50 | 50.65 | 48.30 | 1.67 | 0.00 | 0.00 | 22.55 |
| 3 | GPTQ | 10.33 | 41.13 | 64.44 | 69.34 | 76.12 | 62.75 | 32.15 | 29.90 | 10.20 | 48.25 |
| 4 | GPTQ | 8.41 | 49.23 | 75.29 | 72.92 | 78.24 | 67.80 | 70.51 | 46.50 | 16.60 | 59.64 |
