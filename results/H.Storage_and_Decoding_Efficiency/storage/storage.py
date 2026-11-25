"""Storage of one checkpoint serving all modes of each method, Llama-3.1-8B-Instruct, 224 decoder linear layers (paper: storage table).
Each format's size follows from its tensors; the formulas are first checked byte for byte against the checkpoints in checkpoint_bytes.json
(configurations available locally), then evaluated for the configurations of the table.   usage: python storage.py  (writes storage.json)"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SH = [(4096, 4096), (1024, 4096), (1024, 4096), (4096, 4096), (14336, 4096), (14336, 4096), (4096, 14336)]   # q k v o gate up down (rows x columns)
N = 32 * sum(m * k for m, k in SH); ROWS = 32 * sum(m for m, _ in SH); COLS = 32 * sum(k for _, k in SH); G = 128
def ap(parent, modes):    # parent-bit planes + a table of 2^b fp16 values per row for every mode b
    return {"qweight": N * parent // 8, **{f"lut{b}": ROWS * 2 ** b * 2 for b in modes}}
def ab(modes):            # max(modes) binary planes + per group: b fp16 scales alpha and one offset beta per mode, one shared q_bias
    return {"qweight": N * max(modes) // 8, **{f"alpha_{b}": N // G * b * 2 for b in modes}, **{f"beta_{b}": N // G * 2 for b in modes}, "q_bias": N // G * 2}
def mg():                 # 8-bit master + fp16 scale and zero per group + int64 act_order permutation per column
    return {"qweight": N, "scale": N // G * 2, "zero": N // G * 2, "perm": COLS * 8}
def ar():                 # ladder UHHHHHH (U plane 1 bit, six H planes of 4-bit messages per 8 columns) + 8-bit table of 128 entries per row
    lad = {"ar.plane0": N // 8, **{f"ar.plane{p}": N // 16 for p in range(1, 7)}, "ar.table_codes": ROWS * 128, "ar.table_lo": ROWS * 2, "ar.table_step": ROWS * 2}
    return lad, {"ar.res_planes": N * 4 // 8, "ar.res_lev": ROWS * 30 * 2}   # residual for 5-8 bits: four planes + 30 fp16 offsets per row
meas = json.load(open(os.path.join(HERE, "checkpoint_bytes.json")))
lad, res = ar()
for name, f in (("AP, 4-bit parent (seed 2)", ap(4, (2, 3, 4))), ("AB, 2-4 bits", ab((2, 3, 4))), ("MG, 8-bit master", mg()),
                ("AR (ours), 2-8 bits (released checkpoint Llama-3.1-8B-Instruct-AR2to8)", {**lad, **res})):
    assert f == meas[name], (name, f, meas[name])
print("formulas match the checkpoint files byte for byte")
bpw = lambda d: 8 * sum(d.values()) / N
table = {"AP": {"optimized_bits": "2-8", "modes": 7, "bits_per_weight": bpw(ap(8, range(2, 9)))},
         "AB": {"optimized_bits": "2-8", "modes": 7, "bits_per_weight": bpw(ab(range(2, 9)))},
         "MG (3/4/8)": {"optimized_bits": "3, 4, 8", "modes": "7 (sliced)", "bits_per_weight": bpw(mg())},
         "AR (ours)": {"optimized_bits": "2-4 (by 0.5), 5-8", "modes": 9, "bits_per_weight": bpw({**lad, **res}), "ladder_only_bits_per_weight": bpw(lad)}}
json.dump({"linear_weights": N, "rows": ROWS, "table": table}, open(os.path.join(HERE, "storage.json"), "w"), indent=1)
for m, v in table.items(): print(f"{m:12s} {v['bits_per_weight']:.3f}" + (f"  (ladder {v['ladder_only_bits_per_weight']:.3f})" if "ladder_only_bits_per_weight" in v else ""))
