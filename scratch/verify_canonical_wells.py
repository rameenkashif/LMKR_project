import json, numpy as np

with open('frontend/public/seismic_raw_meta.json') as f:
    meta = json.load(f)

I_len, J_len, K_len = 245, 252, 313
iMin, jMin = 382, 46

with open('frontend/public/seismic_raw.bin', 'rb') as f:
    raw = np.frombuffer(f.read(), dtype=np.int16).reshape((I_len, J_len, K_len))

wells = [
    ('Z-04', 106, 110, 488, 156),
    ('Z-08-ST-02', 38, 110, 420, 156),
    ('Z-02', 153, 147, 535, 193),
    ('Z-03', 46, 100, 428, 146),
    ('Z-05', 16, 153, 398, 199),
    ('Z-06', 63, 162, 445, 208),
    ('Z-07', 106, 153, 488, 199)
]

print("=== VERIFYING EXACT CANONICAL WELL INLINE & CROSSLINE MAPPING ===")
for name, il_local, xl_local, il_real, xl_real in wells:
    trace = np.abs(raw[il_local, xl_local, :])
    tot_energy = trace.sum()
    nz_samples = np.count_nonzero(trace)
    print(f"Well {name}: Inline {il_real} (local #{il_local}), XL {xl_real} (local #{xl_local}) -> Energy={tot_energy}, active samples={nz_samples}/313")
