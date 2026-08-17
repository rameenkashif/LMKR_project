import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ==========================================
# USER CONFIGURATION: Set your output directory name here
# ==========================================
OUTPUT_FOLDER_NAME = "ml_outputs_v11"

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
analysis_dir = os.path.abspath(os.path.join(script_dir, ".."))
training_table_path = os.path.join(analysis_dir, "training_data", "output", "training_table.csv")
output_dir = os.path.join(analysis_dir, OUTPUT_FOLDER_NAME)
facies_dir = os.path.join(output_dir, "facies_model")
os.makedirs(facies_dir, exist_ok=True)

BLIND_WELL = "Z-04"

def main():
    print("--- Phase 3a: Training Sand/Shale Facies Classifier ---")
    if not os.path.exists(training_table_path):
        raise FileNotFoundError(f"Training table not found at {training_table_path}")

    df = pd.read_csv(training_table_path)
    
    # Feature columns and target column
    feature_cols = sorted([c for c in df.columns if c.startswith("attr_")])
    # Limit time-shift window to prevent alignment confusion
    feature_cols = [c for c in feature_cols if not any(x in c for x in ["+3", "+4", "+5", "-3", "-4", "-5"])]
    target_col = "target_IS_SAND"
    
    # Extract unique wells
    wells = df["well_name"].unique()
    print(f"Wells in dataset: {wells}")
    print(f"Number of attributes: {len(feature_cols)}")
    
    # Train / test split (Z-04 held out as blind)
    df_train = df[df["well_name"] != BLIND_WELL].copy()
    df_blind = df[df["well_name"] == BLIND_WELL].copy()
    
    # Standardize features
    scaler = StandardScaler()
    X_train_raw = df_train[feature_cols].values
    y_train = df_train[target_col].values
    
    X_train = scaler.fit_transform(X_train_raw)
    
    # Leave-One-Well-Out Cross-Validation on training wells
    train_wells = [w for w in wells if w != BLIND_WELL]
    cv_scores = []
    cv_f1s = []
    
    print("\nStarting Leave-One-Well-Out Cross-Validation:")
    for test_well in train_wells:
        w_train = df_train[df_train["well_name"] != test_well]
        w_val = df_train[df_train["well_name"] == test_well]
        
        if w_val.empty:
            continue
            
        X_w_train = scaler.transform(w_train[feature_cols].values)
        y_w_train = w_train[target_col].values
        
        X_w_val = scaler.transform(w_val[feature_cols].values)
        y_w_val = w_val[target_col].values
        
        # Train Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        model.fit(X_w_train, y_w_train)
        
        preds = model.predict(X_w_val)
        acc = accuracy_score(y_w_val, preds)
        f1 = f1_score(y_w_val, preds)
        cv_scores.append(acc)
        cv_f1s.append(f1)
        print(f"  Hold-out well {test_well} -> Val Accuracy: {acc:.4f} | Val F1-score: {f1:.4f}")
        
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} | Mean CV F1-score: {np.mean(cv_f1s):.4f}")
    
    # Train final classifier on all training wells
    print("\nTraining final classifier on all training wells...")
    final_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    final_model.fit(X_train, y_train)
    
    # Save components
    joblib.dump(scaler, os.path.join(facies_dir, "facies_scaler.joblib"))
    joblib.dump(final_model, os.path.join(facies_dir, "facies_model.joblib"))
    joblib.dump(feature_cols, os.path.join(facies_dir, "facies_features.joblib"))
    print(f"Saved final facies classifier to: {facies_dir}")
    
    # Evaluate on True Blind Well Z-04
    if not df_blind.empty:
        X_blind = scaler.transform(df_blind[feature_cols].values)
        y_blind = df_blind[target_col].values
        
        preds_blind = final_model.predict(X_blind)
        acc_blind = accuracy_score(y_blind, preds_blind)
        f1_blind = f1_score(y_blind, preds_blind)
        
        print(f"\n=======================================================")
        print(f"BLIND TEST EVALUATION ON WELL {BLIND_WELL} (FACIES CLASSIFIER)")
        print(f"=======================================================")
        print(f"Blind Well Accuracy: {acc_blind:.4f}")
        print(f"Blind Well F1-score: {f1_blind:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_blind, preds_blind))
        print("\nClassification Report:")
        print(classification_report(y_blind, preds_blind, target_names=["Shale", "Sand"], zero_division=0))
        
        # Save validation plot of facies (True vs Predicted over time)
        times = df_blind["seismic_time_ms"].values
        plt.figure(figsize=(4, 8))
        plt.step(y_blind, times, label="Actual Sand/Shale", color="black", where="mid", linewidth=1.5)
        plt.step(preds_blind, times, label="Predicted Facies", color="tab:green", linestyle="--", where="mid", linewidth=1.5)
        plt.gca().invert_yaxis()
        plt.title(f"Blind Well Facies Prediction: {BLIND_WELL}", weight="bold")
        plt.xlabel("Facies (0 = Shale, 1 = Sand)")
        plt.ylabel("Two-Way Time (ms)")
        plt.legend(loc="lower left")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        
        frontend_plot_path = os.path.abspath(os.path.join(analysis_dir, "frontend", "public", "well_images", "facies_blind_well_test.png"))
        plt.savefig(frontend_plot_path, dpi=150)
        v5_plot_path = os.path.join(output_dir, "facies_blind_well_test.png")
        plt.savefig(v5_plot_path, dpi=150)
        plt.close()
        print(f"Saved blind well facies crossplot to: {v5_plot_path}")

if __name__ == "__main__":
    main()
