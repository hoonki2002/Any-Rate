# Encoding

`python -m any_rate.encode --model MODEL` encodes the linear layers of a model and writes a Hugging Face checkpoint folder in the
Any-Rate format. This page describes what the encoder does; the settings are those of the paper (Section 5.1 and Appendices E and G).

## Procedure

For each layer (decoder layer, vision layer, or diffusion block):

1. **Hessian.** Calibration inputs pass through the original model. Forward pre-hooks collect H = sum x x^T of the inputs of every
   group of linear layers that share an input (for example q/k/v, or gate/up). For decoder models the calibration activations are
   pushed through the original layer, so the quantization error of earlier layers never enters later Hessians.
2. **Ladder (2-4 bits).** Every linear layer is encoded with the coded ladder UHHHHHH (`core/ladder.py`): one free plane (U) and six
   Hamming [8,4,4]-coded planes (H), read as the modes 2, 2.5, 3, 3.5 and 4 bits. Columns are processed left to right in blocks of 8 with
   GPTQ-style error feedback and one target per mode; the codewords of a block are chosen by a prefix beam search and the best
   candidates are evaluated exactly. After each pass the shared table (128 entries per row, stored in 8 bits) is refit in closed form;
   the best of the refit rounds is kept.
3. **Residual (5-8 bits).** Four uncoded residual planes and 30 offsets per row correct the 4-bit weight (`core/residual.py`).
4. **Packing.** The indices, the table and the residual are packed into the checkpoint tensors (`core/pack.py`, format below).

## Settings

| setting | value |
|---|---|
| layout | UHHHHHH (modes 2, 2.5, 3, 3.5, 4 bits; 5, 6, 7 and 8 bits from the residual) |
| mode weights | 1-1-2-3-4 |
| prefix beam B, exactly evaluated candidates K | 64, 64 |
| table refit rounds N_ref | 4 |
| damping | 0.01 x mean diagonal of H (0.1, then 1.0 if the damped H is not positive definite) |
| table | one shared table per row, 128 entries, 8-bit codes with fp16 offset and step |
| rows encoded together | 20000 (the best refit round is chosen per group of rows) |
| residual | 4 planes, offsets initialised from the 95th percentile of the residual, refit once |

## Calibration

| --model | encoded | calibration |
|---|---|---|
| `qwen2.5-3b`, `llama-3.1-8b-instruct`, `phi-4` | decoder linear layers | C4, MatGPTQ-style: the first documents of training shard 0 joined with blank lines, 128 x 2048 tokens |
| `gemma-3-4b-it` | decoder linear layers | same C4 set; every sequence starts with `<bos>`; bf16 forward |
| | SigLIP vision tower | the first 128 images of COCO val2017 |
| `qwen3-30b-a3b-instruct-2507` | routed experts (attention, router and norms stay bf16) | same C4 set; one Hessian per expert from the tokens routed to it; an expert with n < 512 tokens gets (512 / n) (tr H / d) I added, one with no token an identity Hessian |
| `whisper-large-v3` | encoder and decoder linear layers | the first 128 utterances of LibriSpeech dev-clean, decoder fed with the transcripts (teacher forcing) |
| `pixart-sigma` | diffusion transformer | 128 prompts of Gustavosta/Stable-Diffusion-Prompts, 20 denoising steps, guidance 4.5 |
| `cogvideox-2b` | diffusion transformer | 32 prompts (`data/cogvideox_calibration_prompts.json`), 49 frames, 50 steps, guidance 6, 4096 random tokens per call |
| `pixart-sigma`, `cogvideox-2b` | T5 text encoder | the 128 Stable-Diffusion prompts, padded to 300 tokens (padding excluded) |

Embeddings, normalization layers, convolutions, the output head and the VAE are not encoded. The calibration data are downloaded
from the Hugging Face Hub on the first run.

## Options

- `--base FOLDER`: a local copy of the original checkpoint instead of the download (for diffusion models, the pipeline folder).
- `--out FOLDER`: output folder (default `<original model name>-AR2to8`). Encoded layers are kept in `<out>/_work`, so an interrupted
  run resumes where it stopped.
- `--layers 0 1 ...`: encode only those layer (block) indices and write no checkpoint.

Llama-3.1 and Gemma-3 are gated on the Hugging Face Hub: accept their licenses and run `huggingface-cli login` first.

The model stays in CPU memory and one layer at a time is moved to the GPU. Peak GPU memory and time on one A100: Qwen2.5-3B 12 GB,
about 2 h; Llama-3.1-8B-Instruct 21 GB, 5 h; Phi-4 28 GB, 11 h; Qwen3-30B-A3B 7 GB (about 64 GB of CPU memory), 32 h.

## Checkpoint format

Each encoded linear layer `<name>` is stored instead of `<name>.weight` as

| tensor | shape, dtype | content |
|---|---|---|
| `<name>.ar.plane0` | (rows, cols/8) uint8 | U plane, column j of a byte = bit j |
| `<name>.ar.plane1` ... `plane6` | (rows, cols/16) uint8 | H planes: 4-bit codeword number per block of 8 columns, two per byte (low nibble first) |
| `<name>.ar.table_codes`, `table_lo`, `table_step` | (rows, 128) uint8, (rows, 1) fp16 x 2 | table value = lo + code x step |
| `<name>.ar.res_planes` | (4, rows, cols/8) uint8 | residual planes, most significant bit first |
| `<name>.ar.res_lev` | (rows, 30) fp16 | residual offsets of the prefixes of 1, 2, 3, 4 residual bits |

The mode that reads the first K ladder planes (K = 3..7) looks up the mean of the 2^(7-K) children of the stored table; the 5- to
8-bit modes add the offset of the first 1 to 4 residual bits to the 4-bit weight (`format.py`). `any_rate_config.json` in the
checkpoint lists the encoded layers and the calibration data.

## Files

    encode.py        command line: models, download, resume, export of the checkpoint folder
    core/ladder.py   coded ladder: Hamming code, prefix beam, per-mode targets, shared table refit, 8-bit table
    core/residual.py residual planes for the 5- to 8-bit modes
    core/pack.py     one linear layer -> checkpoint tensors (rows encoded together, damping fallback)
    calib.py         C4 calibration set, Hessian hooks
    models/          decoder.py (Qwen2.5, Llama-3.1, Phi-4, Gemma-3, Qwen3-MoE), siglip.py, whisper.py, diffusion.py (PixArt-Sigma, CogVideoX-2B)
    format.py        checkpoint format and decoding of a mode
    hf.py            loading a checkpoint with the weights of one mode
