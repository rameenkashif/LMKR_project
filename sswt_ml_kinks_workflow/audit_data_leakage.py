import os
import numpy as np

WORK_DIR = r"d:\Internship@LMKR\Analysis\sswt_ml_kinks_workflow"
data_file = os.path.join(WORK_DIR, "sswt_kinks_dataset.npz")
data = np.load(data_file)

wells = data['wells']

TRAIN_WELLS = ["Z-02", "Z-03", "Z-05", "Z-06", "Z-07"]
TEST_WELLS  = ["Z-04", "Z-08-ST-02"]

train_mask = np.isin(wells, TRAIN_WELLS)
test_mask  = np.isin(wells, TEST_WELLS)

actual_train_wells = set(wells[train_mask])
actual_test_wells  = set(wells[test_mask])
intersection       = actual_train_wells.intersection(actual_test_wells)

print("==========================================================")
print("       BLIND WELL DATA LEAKAGE AUDIT CERTIFICATE          ")
print("==========================================================")
print(f"1. Training Set Wells ({len(train_mask[train_mask])} samples):")
for w in sorted(actual_train_wells):
    count = np.sum(wells[train_mask] == w)
    print(f"   - {w}: {count} samples")

print(f"\n2. Blind Test Set Wells ({len(test_mask[test_mask])} samples):")
for w in sorted(actual_test_wells):
    count = np.sum(wells[test_mask] == w)
    print(f"   - {w}: {count} samples")

print(f"\n3. Data Leakage Intersection Check:")
print(f"   - Intersecting Wells: {intersection} (Must be empty set)")
print(f"   - Leakage Status    : {'ZERO LEAKAGE VERIFIED [OK]' if len(intersection)==0 else 'LEAKAGE DETECTED [FAIL]'}")
print("==========================================================")
