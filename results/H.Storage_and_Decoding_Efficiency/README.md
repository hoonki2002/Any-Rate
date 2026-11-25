# Any-Rate: efficiency results

Kernel latency, decoding throughput, estimated energy and storage of Llama-3.1-8B-Instruct, behind the energy and decoding speed results
and the storage and decoding efficiency appendix of the paper. Measurements on an RTX 5070 Ti (16 GB), batch 1.

    kernel_latency/gemv_llama.json   GEMV time of each linear layer per method and bit width (median of 5 rounds, with spread)
    kernel_latency/validation.json   error of each baseline kernel against its own dense path
    kernel_latency/tables.py         per-layer and per-token tables (tables.md)
    throughput/tps_llama.json        decoding throughput per method and bit width (median of 6 runs, with all runs)
    throughput/table.py              throughput table (table.md)
    energy/energy.py                 bits read per weight, energy per token, energy saved at equal accuracy (energy.json; accuracy from ../A.Full_Benchmark_Results)
    storage/storage.py               bits per weight of one checkpoint serving all modes (storage.json), checked against checkpoint_bytes.json

Methods: AP, AB, MG (MG+perm: with the act_order permutation), AR (ours), FP16.
