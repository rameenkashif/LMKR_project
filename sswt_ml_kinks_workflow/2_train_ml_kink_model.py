import os
import numpy as np
import joblib
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import r2_score, mean_absolute_error, classification_report

WORK_DIR = r"d:\Internship@LMKR\Analysis\sswt_ml_kinks_workflow"
data_file = os.path.join(WORK_DIR, "sswt_kinks_dataset.npz")
data = np.load(data_file)

X = data['X']
y_ai = data['y_ai']
y_kinks = data['y_kinks']
wells = data['wells']

# Instructor Request: 5 Wells for Training, 2 Wells for Testing
TRAIN_WELLS = ["Z-02", "Z-03", "Z-05", "Z-06", "Z-07"]
TEST_WELLS  = ["Z-04", "Z-08-ST-02"]

train_mask = np.isin(wells, TRAIN_WELLS)
test_mask  = np.isin(wells, TEST_WELLS)

X_train, y_ai_train, y_kinks_train = X[train_mask], y_ai[train_mask], y_kinks[train_mask]
X_test,  y_ai_test,  y_kinks_test  = X[test_mask],  y_ai[test_mask],  y_kinks[test_mask]

print(f"Dataset Split:")
print(f" - Train Set (5 Wells: {', '.join(TRAIN_WELLS)}): {X_train.shape[0]} samples")
print(f" - Test Set  (2 Wells: {', '.join(TEST_WELLS)}):  {X_test.shape[0]} samples")

# 1. Train Acoustic Impedance Regressor on SSWT Frequency Features
print("\n[1/2] Training XGBoost Regressor for Acoustic Impedance (AI)...")
ai_model = XGBRegressor(
    n_estimators=250,
    learning_rate=0.04,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42
)
ai_model.fit(X_train, y_ai_train)

# Evaluate Impedance Regressor on Blind Test Wells
y_ai_pred = ai_model.predict(X_test)
r2_ai = r2_score(y_ai_test, y_ai_pred)
mae_ai = mean_absolute_error(y_ai_test, y_ai_pred)

print(f"  [OK] Blind Test Performance (2 Test Wells: Z-04 & Z-08-ST-02):")
print(f"    - R2 Score : {r2_ai:.4f}")
print(f"    - MAE Error: {mae_ai:.2f} (g/cm3 * m/s)")

# 2. Train Boundary Kink Classifier on SSWT Frequency Features
print("\n[2/2] Training XGBoost Classifier for Boundary Kink Detection...")
kink_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    scale_pos_weight=4.0,
    random_state=42
)
kink_model.fit(X_train, y_kinks_train)

# Evaluate Kink Classifier on Blind Test Wells
y_kinks_pred = kink_model.predict(X_test)
print("\n  [OK] Blind Test Kink Detection Report:")
print(classification_report(y_kinks_test, y_kinks_pred, target_names=['No Kink', 'Boundary Kink']))

# Save trained models to workflow folder
ai_model_path = os.path.join(WORK_DIR, "sswt_ai_regressor.joblib")
kink_model_path = os.path.join(WORK_DIR, "sswt_kink_classifier.joblib")

joblib.dump(ai_model, ai_model_path)
joblib.dump(kink_model, kink_model_path)

print(f"\nModels successfully saved to:")
print(f" - {ai_model_path}")
print(f" - {kink_model_path}")
