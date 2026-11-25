"""Whisper-large-v3 WER table, the sentence-level numbers of the text, and the two example utterances.  <set>/<method>/<bits>_summary.json: WER after Whisper's text
normalization; <bits>_samples.jsonl: id, reference, hypothesis of each utterance (greedy, English transcription).
usage: python wer.py   (needs transformers, for Whisper's text normalizer)"""
import json, os
from transformers import WhisperTokenizer
HERE = os.path.dirname(os.path.abspath(__file__)); norm = WhisperTokenizer.from_pretrained("openai/whisper-large-v3").normalize
SETS = ["librispeech_test-clean", "librispeech_test-other", "earnings22"]
def rows(s, m, b): return {r["id"]: r for r in map(json.loads, open(f"{HERE}/{s}/{m}/{b}_samples.jsonl"))}
def wer(s, m, b): return 100 * json.load(open(f"{HERE}/{s}/{m}/{b}_summary.json"))["wer"]
print("WER (%): " + " / ".join(SETS) + " / mean")
for m, bs in (("FP16", ["fp16"]), ("AR_ours", ["8bit", "6bit", "4bit", "3.5bit", "3bit", "2.5bit", "2bit"]), ("MG_2-3-4", ["4bit", "3bit", "2bit"])):
    for b in bs: w = [wer(s, m, b) for s in SETS]; print(f"  {m:9s} {b:6s} " + " ".join(f"{x:6.2f}" for x in w) + f"   {sum(w) / 3:6.2f}")
exact = lambda r: norm(r["hyp"]).strip() == norm(r["ref"]).strip()
a = g = n = 0
for s in SETS:
    A, M = rows(s, "AR_ours", "3bit"), rows(s, "MG_2-3-4", "3bit"); n += len(A)
    a += sum(exact(A[k]) and not exact(M[k]) for k in A); g += sum(exact(M[k]) and not exact(A[k]) for k in A)
print(f"3 bits, exact transcriptions among {n} utterances: Any-Rate only {a}, MG (2/3/4) only {g}")
tc = "librispeech_test-clean"; M2 = rows(tc, "MG_2-3-4", "2bit"); A2 = rows(tc, "AR_ours", "2bit")
print(f"MG (2/3/4) 2-bit, test-clean transcribed as 'Thank you.': {sum(r['hyp'].strip() == 'Thank you.' for r in M2.values())} of {len(M2)}")
for name, R in (("MG (2/3/4)", M2), ("Any-Rate", A2)):
    long_ = [r for r in R.values() if len(r["ref"].split()) >= 5]
    print(f"{name} 2-bit, test-clean utterances of at least five words shortened to two words or fewer: {sum(len(r['hyp'].split()) <= 2 for r in long_)} of {len(long_)}")
print("\nTwo test-other utterances at each bit width:")
for k in ("8280-266249-0022", "6432-63722-0024"):
    print("  reference:", rows("librispeech_test-other", "FP16", "fp16")[k]["ref"])
    for m, bs in (("FP16", ["fp16"]), ("AR_ours", ["4bit", "3.5bit", "3bit", "2.5bit", "2bit"]), ("MG_2-3-4", ["4bit", "3bit", "2bit"])):
        for b in bs: print(f"    {m:9s} {b:6s} {rows('librispeech_test-other', m, b)[k]['hyp']}")
