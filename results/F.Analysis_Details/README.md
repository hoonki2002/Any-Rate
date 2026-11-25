# Any-Rate: analysis

Layer-loss measurements behind the Analysis section and the Analysis Details appendix: 246 matrices of Qwen2.5-3B (first 256 rows, C4 Hessians).

    analysis_numbers.py   recomputes every number of the section and the appendix (tables and text) from data/
    data/per_matrix/      per-matrix results: scalar and coded ladders with and without feedback (nest_feedback_gain_mg),
                          MatGPTQ checkpoints (mg_compare_all, mg_compare124_all, mg_compare348_all), mean propagation (nest_meanprop_mg),
                          the exact two-mode block model on 12 matrices (block_model)
    data/                 predicted gains, column-sorting results, the paper-ladder and damping comparisons, the Gaussian certification,
                          the WikiText-2 PPL of the MatGPTQ (2/3/4) checkpoints with weights 1:1:1 and 1:2:4 (mg_weights_ppl.json)
