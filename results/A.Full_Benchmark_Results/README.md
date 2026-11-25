# Any-Rate: main results

WikiText-2 perplexity and eight benchmarks behind the main results and the full benchmark tables of the paper,
for Qwen2.5-3B-base, Gemma-3-4b-it, Llama-3.1-8B-Instruct, and Phi-4.

    <Model>/results_table.{csv,md}                     the table of the model
    <Model>/per_item_grading/<method>/<bits>/<task>/   the run behind each cell
        <bits>_summary.json                            scores of the run
        <bits>_samples.jsonl                           per-question outputs and grading
    scripts/verify_tables.py                           recomputes the tables from per_item_grading/
    SHA256SUMS

Methods: AP, MG_3-4-8, MG_2-3-4, AB, AR_ours, FP16. Tasks: ppl, csr, gsm8k, triviaqa, nqopen.
