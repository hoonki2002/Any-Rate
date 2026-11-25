# Any-Rate: beyond language models

Outputs of four non-language models encoded by Any-Rate, at each bit width, with MG (2/3/4) where the paper compares it.

    Gemma-3-4b-it_vision/responses.json   responses to "Describe this image in one sentence." for two photos (language model and vision tower
                                           in the same mode); the photos are released with the code
    Whisper-large-v3/<set>/<method>/       WER (<bits>_summary.json) and per-utterance transcripts (<bits>_samples.jsonl) on LibriSpeech
                                           test-clean, test-other and Earnings-22; wer.py prints the table and the sentence-level numbers
    PixArt-Sigma/images/<prompt>/          generated images, <method>_<bits>bit_<parts>_<prompt>.png; generation.json: prompts, seeds, settings
    CogVideoX-2b/videos/                   generated videos, robot_<method>_<bits>bit_dit+t5.mp4; generation.json: prompt, seed, settings

Methods: fp16 / FP16 (original model), ar / AR_ours (Any-Rate), mg / MG_2-3-4 (MatGPTQ trained for 2/3/4 bits).
