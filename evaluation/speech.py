"""Whisper-large-v3: WER on LibriSpeech test-clean, test-other and Earnings-22 (greedy English transcription, Whisper's text
normalization, utterances in order of length, batch 16). Results: <root>/<set>/<method>/<bits>_summary.json and _samples.jsonl."""
import io, json, os, time
import numpy as np
import torch

SETS = {"librispeech_test-clean": ("openslr/librispeech_asr", ["clean/test/0000.parquet"]),
        "librispeech_test-other": ("hf-audio/esb-datasets-test-only-sorted", ["librispeech/test.other-00000-of-00001.parquet"]),
        "earnings22": ("hf-audio/esb-datasets-test-only-sorted", [f"earnings22/test-0000{i}-of-00005.parquet" for i in range(5)])}


def load_set(name):
    """16 kHz int16 audio, transcripts and ids"""
    import pyarrow.parquet as pq, soundfile as sf
    from huggingface_hub import hf_hub_download
    repo, files = SETS[name]; audio, text, ids = [], [], []
    for fn in files:
        for r in pq.read_table(hf_hub_download(repo, fn, repo_type="dataset")).to_pylist():
            if repo.startswith("openslr"):                              # 16 kHz int16 FLAC
                audio.append(sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="int16")[0].copy()); text.append(r["text"]); ids.append(r["id"]); continue
            a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")   # some sets store float WAV or other rates
            if a.ndim > 1: a = a[:, 0]
            if sr != 16000:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(16000, sr); a = resample_poly(a, 16000 // g, sr // g)
            audio.append(np.clip(np.round(a * 32767), -32768, 32767).astype("int16")); text.append(r["text"]); ids.append(str(r.get("id", len(ids))))
    return audio, text, ids


@torch.no_grad()
def run(method, ckpt, base, modes, root, d, limit, log):
    import jiwer
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from . import weights
    key = lambda b: "fp16" if method == "fp16" else f"{b:g}bit"
    todo = [b for b in modes if any(not os.path.exists(os.path.join(root, s, d, f"{key(b)}_summary.json")) for s in SETS)]
    if not todo: return
    proc = WhisperProcessor.from_pretrained(base); model = WhisperForConditionalGeneration.from_pretrained(base, torch_dtype=torch.float16).cuda().eval()
    norm = proc.tokenizer.normalize; data = {s: load_set(s) for s in SETS}
    for b in todo:
        if method != "fp16": log(f"{d} {key(b)}: {weights.apply(model, weights.of(method, ckpt, b))} layers set")
        for s, (audio, text, ids) in data.items():
            out = os.path.join(root, s, d)
            if os.path.exists(os.path.join(out, f"{key(b)}_summary.json")): continue
            order = sorted(range(len(audio)), key=lambda i: len(audio[i]))[:limit]; hyp = {}; t0 = time.time()
            for st in range(0, len(order), 16):
                ix = order[st:st + 16]
                feats = proc.feature_extractor([audio[i].astype("float32") / 32768.0 for i in ix], sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
                gen = model.generate(input_features=feats, language="en", task="transcribe", num_beams=1, do_sample=False, max_new_tokens=200)
                for i, t in zip(ix, proc.batch_decode(gen, skip_special_tokens=True)): hyp[i] = t.strip()
            keep = [i for i in order if norm(text[i]).strip()]
            wer = jiwer.wer([norm(text[i]) for i in keep], [norm(hyp[i]) for i in keep]); os.makedirs(out, exist_ok=True)
            json.dump({"wer": wer, "n": len(keep)}, open(os.path.join(out, f"{key(b)}_summary.json"), "w"))
            with open(os.path.join(out, f"{key(b)}_samples.jsonl"), "w") as f:
                for i in order: f.write(json.dumps({"id": ids[i], "ref": text[i], "hyp": hyp[i]}) + "\n")
            log(f"{d} {key(b)} {s}: WER {100 * wer:.2f}% ({len(keep)} utterances, {time.time() - t0:.0f}s)")
