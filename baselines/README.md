# Baselines

The baselines of the paper, encoded with their authors' code: the multi-precision methods AP, AB and MG, and single-precision GPTQ (one model
per bit width). `setup.sh` clones each repository at the commit we used and applies a small patch to the first three (model support
only, see [Patches](#patches)). This folder is not affiliated with or endorsed by the
authors of these methods; please cite their papers (below) when you use their code.

| | method | code (commit) | license |
|---|---|---|---|
| AP | Any-Precision LLM (Park et al., ICML 2024) | [SNU-ARC/any-precision-llm](https://github.com/SNU-ARC/any-precision-llm) (`ee52889`) | MIT |
| AB | AnyBCQ (Park et al., ICLR 2026) | [naver-aics/anybcq](https://github.com/naver-aics/anybcq) (`c316a4d`) | Apache-2.0 |
| MG | MatGPTQ (Kleinegger et al., 2026) | [IST-DASLab/MatGPTQ](https://github.com/IST-DASLab/MatGPTQ) (`86ee3ee`) | MIT |
| GPTQ | GPTQ (Frantar et al., ICLR 2023), single-precision | [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq) (`2d65066`) | Apache-2.0 |

## Setup

```
python -m pip install -r baselines/requirements.txt
bash baselines/setup.sh
```

AnyBCQ, MatGPTQ and GPTQ run in the Any-Rate environment. Any-Precision LLM needs older numpy and datasets, so `setup.sh` gives it its own
environment (`baselines/third_party/venv_ap`, from `any_precision/requirements.txt`).

## Encoding

```
python -m baselines.any_precision.encode --model qwen2.5-3b
python -m baselines.anybcq.encode --model qwen2.5-3b
python -m baselines.matgptq.encode --model qwen2.5-3b --recipe 234
python -m baselines.matgptq.encode --model qwen2.5-3b --recipe 348
python -m baselines.gptq.encode --model qwen2.5-3b
```

The original checkpoint is downloaded (`--base`: a local copy) and the output is written to `<original model name>-<method>`
(`--out`: another folder). `dense.py` of each method reads one mode of its output as dense weights.

| method | recipe | `--model` |
|---|---|---|
| AP | authors' defaults with a 2-bit seed: 8-bit parent, gradients on C4 (100 x 512 tokens); modes 2-8 | `qwen2.5-3b`, `gemma-3-4b-it`, `llama-3.1-8b-instruct`, `phi-4` |
| AB | authors' PTQ recipe, 2 -> 4 bits: 512 C4 samples, 5000 iterations, group 128, symmetric, input_prob 0.5 (Llama, Gemma) or 1.0 (Qwen, Phi); TF32 | same |
| MG (3/4/8) | authors' official recipe: 8-bit master, slices 3/4/8 with weights 1/2/4, group 128, act-order, scale search; C4 128 x 2048 tokens | same |
| MG (2/3/4) | the same with a 4-bit master and slices 2/3/4 with weights 1/1/1 | same, and `qwen3-30b-a3b-instruct-2507`, `whisper-large-v3`, `pixart-sigma`, `cogvideox-2b` |
| GPTQ | single-precision, one model per bit width (2, 3, 4): asymmetric, group 128, act-order, damping 0.01, block 128; the authors' GPTQ and Quantizer classes on the Hessians of the Any-Rate calibration (C4 128 x 2048 tokens) | `qwen2.5-3b`, `gemma-3-4b-it`, `llama-3.1-8b-instruct`, `phi-4` |

Gemma-3-4b-it: the baselines quantize its language model, written once as a text-only checkpoint (`gemma-3-4b-it-text`), in bf16.
Qwen3-30B-A3B: MG quantizes only the routed experts, with the rule of the Any-Rate MoE encoding for experts that receive few or no
calibration tokens. Whisper, PixArt-Sigma and CogVideoX-2B: the MatGPTQ class of the authors' code on the Hessians of the Any-Rate
calibration of that model (`any_rate/models`).

## Patches

| patch | change |
|---|---|
| `any_precision/any_precision.patch` | architecture files for Qwen2, Phi-3 and Gemma-3; the cached gradients are memory-mapped |
| `anybcq/anybcq.patch` | Gemma-3: architecture file, and the separate rotary embeddings of its global and sliding-window layers in the block reconstruction |
| `matgptq/matgptq.patch` | model types `qwen2` and `gemma3_text`; fast tokenizer; the MoE expert rule (only with `MOE_PRIOR` set, as `encode.py` does for Qwen3-30B-A3B) |

`launch.py` runs the authors' scripts unchanged; it turns on TF32 for AnyBCQ and stands in for two modules that are imported but not
used in these runs (`custom_kernel`, the AnyBCQ inference kernels, and `wandb`). Every changed line carries the comment
`modified by the Any-Rate authors`, and every added file starts with `Added by the Any-Rate authors`. GPTQ is used unpatched:
`gptq/encode.py` imports its `GPTQ` and `Quantizer` classes.

## License

Each patch follows the license of the code it modifies (`LICENSE` in each folder; `gptq/LICENSE`: the license of the GPTQ code).

## Citation

```bibtex
@inproceedings{park2024anyprecision,
  title     = {{Any-Precision LLM}: Low-Cost Deployment of Multiple, Different-Sized {LLMs}},
  author    = {Park, Yeonhong and Hyun, Jake and Cho, Sanglyul and Sim, Bonggeun and Lee, Jae W.},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024}
}
@inproceedings{park2026anybcq,
  title     = {{AnyBCQ}: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision {LLMs}},
  author    = {Park, Gunho and Bae, Jeongin and Kwon, Beomseok and Kim, Byeongwook and Kwon, Se Jung and Lee, Dongsoo},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
@article{kleinegger2026matgptq,
  title   = {{MatGPTQ}: Efficient and Accurate Inference over Nested Quantized Models},
  author  = {Kleinegger, Maximilian and Crn{\v{c}}evi{\'c}, Elvir and Alistarh, Dan},
  journal = {arXiv preprint arXiv:2602.03537},
  year    = {2026}
}
@inproceedings{frantar2023gptq,
  title     = {{GPTQ}: Accurate Post-Training Quantization for Generative Pre-trained Transformers},
  author    = {Frantar, Elias and Ashkboos, Saleh and Hoefler, Torsten and Alistarh, Dan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023}
}
```
