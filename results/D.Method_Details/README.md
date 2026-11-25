# Any-Rate: method details

The two measured tables of the Method Details appendix, for the paper's Qwen2.5-3B encoding.

    residual_cost.py                         bits per weight of three ways to support 5-8 bits, checked against checkpoint_bytes_qwen25_3b.json
    residual_loss.py                         relative proxy loss of the 4-8-bit modes (mean over 252 matrices)
    residual_proxy_loss_per_matrix.json      per-matrix values behind residual_loss.py
