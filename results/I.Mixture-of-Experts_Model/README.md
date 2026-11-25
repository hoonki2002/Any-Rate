# Any-Rate: mixture-of-experts model

Qwen3-30B-A3B-Instruct-2507 with its routed experts encoded by Any-Rate: accuracy, per-expert mode allocation and decoding speed.

    accuracy/<run>/<bits>/<task>/   <bits>_summary.json and <bits>_samples.jsonl (ppl: summary only) for
                                    BF16, AR_uniform, AR_gate_mass, AR_reverse_gate_mass, AR_attention_only, AR_experts_and_attention, MG_2-3-4 (PPL)
    allocation/                     per-expert modes by gate mass (b<bits>_greedy.json) and in reverse order (b<bits>_reverse.json),
                                    calibration tokens routed to each expert
    speed/                          decoding speed per format and VRAM budget (RTX 5070 Ti, NVMe expert cache)
    moe_results.py                  accuracy table and the numbers of the text

Tasks: ppl, csr, gsm8k, triviaqa, nqopen, scored as in ../A.Full_Benchmark_Results.
