"""Relative proxy loss (%) of the 4-8-bit modes of the paper's Qwen2.5-3B encoding, mean over the 252 linear layers (paper: Method Details,
residual planes). Per-matrix values (fraction, Hessian-weighted error / Hessian-weighted weight energy) in residual_proxy_loss_per_matrix.json."""
import json, os
d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "residual_proxy_loss_per_matrix.json")))
print(f"{len(d)} matrices"); print("| mode (bits) | " + " | ".join(str(k) for k in range(4, 9)) + " |\n|---|---|---|---|---|---|")
print("| relative proxy loss (%) | " + " | ".join(f"{100 * sum(v[f'rel{k}'] for v in d.values()) / len(d):.3f}" for k in range(4, 9)) + " |")
