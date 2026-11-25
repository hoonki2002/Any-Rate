"""Whisper (encoder and decoder linear layers). Calibration: the first 128 utterances of LibriSpeech dev-clean, encoder fed with
log-mel features and the decoder with the reference transcripts (teacher forcing); padding positions are excluded from the Hessians.
The Hessians of `per_pass` layer indices (both stacks) are collected in one pass over the calibration set."""
import io
import torch
from ..calib import HessianHooks

ENC = [("e_qkv", "self_attn.q_proj", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], None), ("e_o", "self_attn.out_proj", ["self_attn.out_proj"], None),
       ("e_fc1", "fc1", ["fc1"], None), ("e_fc2", "fc2", ["fc2"], None)]
DEC = [("d_qkv", "self_attn.q_proj", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "dmask"), ("d_o", "self_attn.out_proj", ["self_attn.out_proj"], "dmask"),
       ("d_cq", "encoder_attn.q_proj", ["encoder_attn.q_proj"], "dmask"), ("d_ckv", "encoder_attn.k_proj", ["encoder_attn.k_proj", "encoder_attn.v_proj"], None),
       ("d_co", "encoder_attn.out_proj", ["encoder_attn.out_proj"], "dmask"), ("d_fc1", "fc1", ["fc1"], "dmask"), ("d_fc2", "fc2", ["fc2"], "dmask")]


def librispeech_dev_clean(n=128):
    import pyarrow.parquet as pq, soundfile as sf
    from huggingface_hub import hf_hub_download
    rows = pq.read_table(hf_hub_download("openslr/librispeech_asr", "clean/validation/0000.parquet", repo_type="dataset")).to_pylist()[:n]
    audio = [torch.from_numpy(sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="int16")[0].copy()) for r in rows]
    return audio, [r["text"] for r in rows]


@torch.no_grad()
def encode(model_dir, encode_layer, save, layers=None, dev="cuda", nsamples=128, per_pass=8, log=print):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(model_dir); model = WhisperForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float16).to(dev).eval(); net = model.model
    tok = proc.tokenizer; tok.set_prefix_tokens(language="en", task="transcribe", predict_timestamps=False)
    audio, text = librispeech_dev_clean(nsamples); todo = sorted(range(len(net.encoder.layers)) if layers is None else layers); g = {}

    def batches():
        for i in range(0, nsamples, 8):
            feats = proc.feature_extractor([a.numpy().astype("float32") / 32768.0 for a in audio[i:i + 8]], sampling_rate=16000, return_tensors="pt").input_features.to(dev, torch.float16)
            ids = [tok(t.lower()).input_ids + [tok.eos_token_id] for t in text[i:i + 8]]; T = max(map(len, ids))
            di = torch.tensor([x + [tok.eos_token_id] * (T - len(x)) for x in ids], device=dev); dm = torch.tensor([[1] * len(x) + [0] * (T - len(x)) for x in ids], device=dev)
            yield feats, di, dm

    for c0 in range(0, len(todo), per_pass):
        units = todo[c0:c0 + per_pass]; mods, masks = {}, {}
        for i in units:
            for gk, hp, names, mk in ENC: mods[("enc", i, gk)] = net.encoder.layers[i].get_submodule(hp); masks[("enc", i, gk)] = mk
            for gk, hp, names, mk in DEC: mods[("dec", i, gk)] = net.decoder.layers[i].get_submodule(hp); masks[("dec", i, gk)] = mk
        hk = HessianHooks(mods, dev, mask=lambda key: g.get(masks[key]) if masks[key] else None)
        for feats, di, dm in batches(): g["dmask"] = dm; model(input_features=feats, decoder_input_ids=di, decoder_attention_mask=dm)
        hk.remove(); H = hk.H
        for k in H: assert torch.isfinite(H[k]).all(), f"non-finite Hessian {k}"
        for i in units:
            store = {}
            for stack, groups in (("enc", ENC), ("dec", DEC)):
                blk = (net.encoder if stack == "enc" else net.decoder).layers[i]
                for gk, hp, names, mk in groups:
                    for nm in names:
                        store[f"model.{'encoder' if stack == 'enc' else 'decoder'}.layers.{i}.{nm}"] = encode_layer(blk.get_submodule(nm).weight.data.float(), H[(stack, i, gk)])
                    del H[(stack, i, gk)]; torch.cuda.empty_cache()
            save(i, store); log(f"layer {i} (encoder and decoder) encoded")
