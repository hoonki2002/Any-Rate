# Llama-3.1-8B-Instruct: single-precision GPTQ

PPL = WikiText-2. Benchmarks in %: ARC-C/ARC-E/HellaSwag/PIQA = acc_norm, WinoGrande = acc, GSM8K = flexible-extract exact match, TriviaQA/NQ-open = exact match. Avg8 = mean of the 8 benchmarks. One model per bit width; FP16, MG (2/3/4) and AR (ours): ../A.Full_Benchmark_Results.

| bit | method | PPL(WikiText-2) | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | GSM8K(flexible) | TriviaQA | NQ-open | Avg8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | GPTQ | 40786.39 | 26.19 | 26.30 | 26.38 | 51.52 | 48.38 | 0.61 | 0.00 | 0.00 | 22.42 |
| 3 | GPTQ | 9.42 | 43.69 | 68.64 | 74.98 | 75.90 | 70.48 | 43.06 | 48.00 | 16.40 | 55.14 |
| 4 | GPTQ | 7.55 | 54.61 | 79.34 | 78.83 | 80.47 | 72.93 | 68.61 | 71.00 | 25.70 | 66.44 |
