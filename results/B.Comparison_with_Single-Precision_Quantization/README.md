# Any-Rate: comparison with single-precision quantization

Single-precision GPTQ (one model per bit width) at 2, 3 and 4 bits for Qwen2.5-3B-base, Gemma-3-4b-it, Llama-3.1-8B-Instruct and Phi-4, encoded
with `baselines/gptq/encode.py` (the authors' code on the Hessians of the Any-Rate calibration) and evaluated as the main results.
FP16, MG (2/3/4) and Any-Rate of the same comparison are in ../A.Full_Benchmark_Results.

    <Model>/results_table.{csv,md}                     the GPTQ rows of the table
    <Model>/per_item_grading/GPTQ/<bits>/<task>/       the run behind each cell
        <bits>_summary.json                            scores of the run
        <bits>_samples.jsonl                           per-question outputs and grading (ppl: summary only)
    scripts/verify_tables.py                           recomputes the tables from per_item_grading/
    scripts/numbers.py                                 the averages over the four models, the per-model accuracy and the storage of the appendix
    SHA256SUMS

Tasks: ppl, csr, gsm8k, triviaqa, nqopen, scored as in ../A.Full_Benchmark_Results.
