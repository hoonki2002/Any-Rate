# Gemma-3-4b-it: single-precision GPTQ

PPL = WikiText-2. Benchmarks in %: ARC-C/ARC-E/HellaSwag/PIQA = acc_norm, WinoGrande = acc, GSM8K = flexible-extract exact match, TriviaQA/NQ-open = exact match. Avg8 = mean of the 8 benchmarks. One model per bit width; FP16, MG (2/3/4) and AR (ours): ../A.Full_Benchmark_Results.

| bit | method | PPL(WikiText-2) | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | GSM8K(flexible) | TriviaQA | NQ-open | Avg8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | GPTQ | 94317.53 | 22.44 | 27.61 | 27.80 | 50.92 | 51.46 | 0.68 | 0.10 | 0.00 | 22.63 |
| 3 | GPTQ | 17.38 | 46.42 | 72.60 | 67.35 | 74.86 | 64.09 | 22.14 | 33.70 | 11.40 | 49.07 |
| 4 | GPTQ | 17.03 | 56.66 | 77.82 | 73.02 | 77.42 | 67.64 | 70.66 | 46.90 | 15.90 | 60.75 |
