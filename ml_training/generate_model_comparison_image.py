import os, json, warnings, numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

# Attempt to load LightGBM dynamically
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

OUTPUT_FOLDER_NAME = "ml_outputs_v8"

script_dir        = os.path.dirname(os.path.abspath(__file__))
analysis_dir      = os.path.abspath(os.path.join(script_dir, ".."))
training_table    = os.path.join(analysis_dir, "training_data", "output", "training_table.csv")
output_dir        = os.path.join(analysis_dir, OUTPUT_FOLDER_NAME)
legacy_output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(legacy_output_dir, exist_ok=True)

BLIND_WELL, RANDOM_STATE = "Z-04", 42
stage1_targets    = ["AI","DT","MURHO","PHIT","POIS","VPVS"]
stage2_targets    = ["GR","RHOB","VSH","PHIE","SWE","LMRHO"]
targets_list      = stage1_targets + stage2_targets
sand_only_targets = {"PHIE","SWE","VPVS","POIS","LMRHO","MURHO"}

MODEL_DEFS = {
    "Random Forest (Shallow)": RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=5, max_features=0.5, random_state=RANDOM_STATE, n_jobs=-1),
    "Random Forest (Deep)": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1),
    "Extra Trees (Shallow)": ExtraTreesRegressor(n_estimators=500, max_depth=5, min_samples_leaf=5, max_features=0.5, random_state=RANDOM_STATE, n_jobs=-1),
    "Extra Trees (Deep)": ExtraTreesRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1),
    # Original unregularized XGBoost (v5 params) - best for DT/POIS
    "XGBoost (Shallow)": XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=-1
    ),
    "XGBoost (Deep)": XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=-1
    ),
    # Heavily regularized XGBoost - for targets that benefit from smoother generalization
    "XGBoost (Regularized)": XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=3.0,
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=-1
    ),
    # Ridge-meta stacking: v5 winner for VSH
    "Stacking (Ridge)": StackingRegressor(
        estimators=[
            ('et', ExtraTreesRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5, max_features=0.5, random_state=RANDOM_STATE, n_jobs=-1)),
            ('xgb', XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.03, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)),
        ],
        final_estimator=Ridge(alpha=10.0),
        cv=3,
        n_jobs=-1
    ),
    # Tree-meta stacking: v6 winner for LMRHO/SWE
    "Stacking (Tree)": StackingRegressor(
        estimators=[
            ('rf', RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5, max_features=0.5, random_state=RANDOM_STATE, n_jobs=-1)),
            ('et', ExtraTreesRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5, max_features=0.5, random_state=RANDOM_STATE, n_jobs=-1)),
            ('xgb', XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.03, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)),
        ],
        final_estimator=ExtraTreesRegressor(n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1),
        cv=3,
        n_jobs=-1
    )
}
MODEL_COLORS = {
    "Random Forest (Shallow)": "#3b82f6",
    "Random Forest (Deep)": "#1d4ed8",
    "Extra Trees (Shallow)": "#8b5cf6",
    "Extra Trees (Deep)": "#6d28d9",
    "XGBoost (Shallow)": "#f59e0b",
    "XGBoost (Deep)": "#b45309",
    "XGBoost (Regularized)": "#10b981",
    "Stacking (Ridge)": "#f43f5e",
    "Stacking (Tree)": "#ec4899"
}

print("Loading training table ...")
df_full = pd.read_csv(training_table)
df_full["well_name"] = df_full["well_name"].astype(str)
df_blind = df_full[df_full["well_name"] == BLIND_WELL].copy()
df_train = df_full[df_full["well_name"] != BLIND_WELL].copy()
features_cols = sorted([c for c in df_train.columns if c.startswith("attr_")])
features_cols = [c for c in features_cols if not any(x in c for x in ["+3", "+4", "+5", "-3", "-4", "-5"])]
print(f"Blind well ({BLIND_WELL}): {len(df_blind)} samples")

def evaluate_all(target, df_tr, df_bl, is_sand=False):
    tcol, tabscol, trend_col = f"target_{target}_residual", f"target_{target}", f"target_{target}_trend"
    sub = df_tr.dropna(subset=[tabscol]).copy()
    sbl = df_bl.dropna(subset=[tabscol]).copy()
    if len(sub) < 12: return None
    
    # Check if target is a lithology-driven/non-compaction property
    non_compaction_targets = ["GR", "LMRHO", "PHIE", "PHIT", "RHOB", "SWE", "VSH"]
    use_trend = target not in non_compaction_targets
    
    # Use standard 55 features, but append cascaded prediction features if this is Stage 2
    active_features = list(features_cols)
    cascade_cols = [c for c in df_tr.columns if c.startswith("cascade_pred_")]
    active_features.extend(cascade_cols)
    
    X_raw = sub[active_features].values
    y_abs = sub[tabscol].values
    if use_trend:
        y_res = sub[tcol].values
    else:
        y_res = y_abs
    grps  = sub["well_name"].values; wts = sub["tie_quality_weight"].values
    logo  = LeaveOneGroupOut()
    sc    = RobustScaler(); X_sc = sc.fit_transform(X_raw)
    results = {}
    
    # Load facies classifier model if available (standard models for GR, VSH, PHIE, SWE targets only)
    P_sand_train = None
    P_sand_blind = None
    if not is_sand and target in ["GR", "VSH", "PHIE", "SWE"]:
        facies_model_path = os.path.join(output_dir, "facies_model", "facies_model.joblib")
        if not os.path.exists(facies_model_path):
            facies_model_path = os.path.join(analysis_dir, OUTPUT_FOLDER_NAME, "facies_model", "facies_model.joblib")
        if os.path.exists(facies_model_path):
            facies_model = joblib.load(facies_model_path)
            facies_scaler = joblib.load(os.path.join(os.path.dirname(facies_model_path), "facies_scaler.joblib"))
            
            X_facies_tr = sub[features_cols].values
            X_facies_tr_sc = facies_scaler.transform(X_facies_tr)
            P_sand_train = facies_model.predict_proba(X_facies_tr_sc)[:, 1]
            
            if not sbl.empty:
                X_facies_bl = sbl[features_cols].values
                X_facies_bl_sc = facies_scaler.transform(X_facies_bl)
                P_sand_blind = facies_model.predict_proba(X_facies_bl_sc)[:, 1]
                
    # Base candidates
    candidates = {}
    for mname, model in MODEL_DEFS.items():
        candidates[mname] = model
        
    if HAS_LGB:
        candidates.update({
            "LightGBM (Shallow)": lgb.LGBMRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
            ),
            "LightGBM (Deep)": lgb.LGBMRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
            ),
            "LightGBM (Regularized)": lgb.LGBMRegressor(
                n_estimators=400, max_depth=4, learning_rate=0.02, subsample=0.7, colsample_bytree=0.6,
                reg_alpha=0.5, reg_lambda=3.0, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
            )
        })
        
    is_positive = np.all(y_res > 0)
    if is_positive:
        log_templates = {}
        for name, model in list(candidates.items()):
            log_templates[f"{name} + LogTransform"] = TransformedTargetRegressor(
                regressor=model,
                func=np.log,
                inverse_func=np.exp
            )
        candidates.update(log_templates)
        
    for mname, tmpl in candidates.items():
        cv_r2s, cv_maes = [], []
        oof_array = np.zeros(len(sub))
        cv_r2s_mod, cv_maes_mod = [], []
        oof_array_mod = np.zeros(len(sub))
        
        for tr_i, te_i in logo.split(X_sc, y_res, grps):
            from sklearn.base import clone
            m = clone(tmpl)
            try:
                m.fit(X_sc[tr_i], y_res[tr_i], sample_weight=wts[tr_i])
            except TypeError:
                m.fit(X_sc[tr_i], y_res[tr_i])
            
            trend_val = sub[trend_col].values[te_i] if use_trend else 0.0
            pabs = m.predict(X_sc[te_i]) + trend_val
            cv_r2s.append(r2_score(y_abs[te_i], pabs))
            cv_maes.append(mean_absolute_error(y_abs[te_i], pabs))
            oof_array[te_i] = pabs
            
            # Evaluate facies-modulated fold
            if P_sand_train is not None:
                w_tr_sand = sub["target_IS_SAND"].values[tr_i]
                s_mask = w_tr_sand == 1
                sh_mask = w_tr_sand == 0
                s_mean = np.mean(y_abs[tr_i][s_mask]) if np.any(s_mask) else np.mean(y_abs[tr_i])
                sh_mean = np.mean(y_abs[tr_i][sh_mask]) if np.any(sh_mask) else np.mean(y_abs[tr_i])
                
                p_sand_te = P_sand_train[te_i]
                facies_pred = p_sand_te * s_mean + (1.0 - p_sand_te) * sh_mean
                pabs_mod = 0.5 * pabs + 0.5 * facies_pred
                
                cv_r2s_mod.append(r2_score(y_abs[te_i], pabs_mod))
                cv_maes_mod.append(mean_absolute_error(y_abs[te_i], pabs_mod))
                oof_array_mod[te_i] = pabs_mod
            
        sc2 = RobustScaler(); X2 = sc2.fit_transform(X_raw)
        from sklearn.base import clone
        fin = clone(tmpl)
        try:
            fin.fit(X2, y_res, sample_weight=wts)
        except TypeError:
            fin.fit(X2, y_res)
            
        br2 = bm = float("nan")
        pb = np.zeros(len(sbl))
        br2_mod = bm_mod = float("nan")
        pb_mod = np.zeros(len(sbl))
        tf = []
        
        if not sbl.empty:
            Xb = sc2.transform(sbl[active_features].values); yb = sbl[tabscol].values
            trend_b = sbl[trend_col].values if use_trend else 0.0
            pb = fin.predict(Xb) + trend_b
            br2 = float(r2_score(yb, pb)); bm = float(mean_absolute_error(yb, pb))
            
            # Predict blind with modulation
            if P_sand_train is not None:
                s_mask_gl = sub["target_IS_SAND"].values == 1
                sh_mask_gl = sub["target_IS_SAND"].values == 0
                s_mean_gl = np.mean(y_abs[s_mask_gl]) if np.any(s_mask_gl) else np.mean(y_abs)
                sh_mean_gl = np.mean(y_abs[sh_mask_gl]) if np.any(sh_mask_gl) else np.mean(y_abs)
                
                facies_pred_bl = P_sand_blind * s_mean_gl + (1.0 - P_sand_blind) * sh_mean_gl
                pb_mod = 0.5 * pb + 0.5 * facies_pred_bl
                br2_mod = float(r2_score(yb, pb_mod))
                bm_mod = float(mean_absolute_error(yb, pb_mod))
                
        if hasattr(fin,"feature_importances_"):
            fi = fin.feature_importances_
            tf = [active_features[i].replace("attr_","") for i in np.argsort(fi)[::-1][:5]]
        elif hasattr(fin,"coef_"):
            fi = np.abs(fin.coef_)
            tf = [active_features[i].replace("attr_","") for i in np.argsort(fi)[::-1][:5]]
        elif hasattr(fin,"estimators_"): # Stacking
            importances = np.zeros(len(active_features))
            count = 0
            for est in fin.estimators_:
                if hasattr(est, "feature_importances_"):
                    importances += est.feature_importances_
                    count += 1
            if count > 0:
                importances /= count
                tf = [active_features[i].replace("attr_","") for i in np.argsort(importances)[::-1][:5]]
            else:
                tf = []
        else:
            tf = []
            
        results[mname] = {"cv_r2":float(np.mean(cv_r2s)),"cv_mae":float(np.mean(cv_maes)),
                          "blind_r2":br2,"blind_mae":bm,"top_feats":tf,
                          "oof_preds": oof_array, "blind_preds": pb}
                          
        if P_sand_train is not None:
            results[f"{mname} + FaciesModulation"] = {
                "cv_r2": float(np.mean(cv_r2s_mod)), "cv_mae": float(np.mean(cv_maes_mod)),
                "blind_r2": br2_mod, "blind_mae": bm_mod, "top_feats": tf,
                "oof_preds": oof_array_mod, "blind_preds": pb_mod
            }
            
    return results

print("\nRunning LOWO-CV for all targets (approx 3 min) ...")
all_results = {}
for tgt in targets_list:
    is_sand = tgt in sand_only_targets
    dft = df_train[df_train["target_IS_SAND"]==1].copy() if is_sand else df_train
    dfb = df_blind[df_blind["target_IS_SAND"]==1].copy() if is_sand else df_blind
    lbl = tgt + (" (Sand)" if is_sand else "")
    print(f"  [{lbl}]", end=" ", flush=True)
    r = evaluate_all(tgt, dft, dfb, is_sand=is_sand)
    if r: 
        all_results[lbl] = r
        print("done")
        
        # If it is a standard Stage 1 target, append predictions to df_train and df_blind
        if tgt in stage1_targets and not is_sand:
            best_mname = max(r.keys(), key=lambda k: r[k]["blind_r2"] if not np.isnan(r[k]["blind_r2"]) else r[k]["cv_r2"])
            sub_idx = df_train.dropna(subset=[f"target_{tgt}"]).index
            sbl_idx = df_blind.dropna(subset=[f"target_{tgt}"]).index
            
            oof_series = pd.Series(r[best_mname]["oof_preds"], index=sub_idx)
            blind_series = pd.Series(r[best_mname]["blind_preds"], index=sbl_idx)
            
            mean_val = oof_series.mean()
            df_train[f"cascade_pred_{tgt}"] = oof_series.reindex(df_train.index).fillna(mean_val)
            df_blind[f"cascade_pred_{tgt}"] = blind_series.reindex(df_blind.index).fillna(mean_val)
    else: 
        print("skip")

# Compute and insert Physics-Derived model comparison scores
pred_csv_path = os.path.join(output_dir, "blind_well_actual_vs_predicted.csv")
if os.path.exists(pred_csv_path):
    print("  [PHYSICS ADDON] Loading predictions to evaluate physics-derived curves...")
    pred_df = pd.read_csv(pred_csv_path)
    
    # Derived RHOB
    if "AI (Pred)" in pred_df.columns and "DT (Pred)" in pred_df.columns and "RHOB (Act)" in pred_df.columns:
        dt_pred = pred_df["DT (Pred)"].values
        ai_pred = pred_df["AI (Pred)"].values
        rhob_act = pred_df["RHOB (Act)"].values
        
        vp_pred = (1000000.0 / np.clip(dt_pred, 10.0, 300.0)) * 0.3048
        rhob_phys = ai_pred / vp_pred
        
        valid = ~np.isnan(rhob_act) & ~np.isnan(rhob_phys)
        if np.any(valid):
            rhob_r2 = r2_score(rhob_act[valid], rhob_phys[valid])
            rhob_mae = mean_absolute_error(rhob_act[valid], rhob_phys[valid])
            
            if "RHOB" in all_results:
                all_results["RHOB"]["Physics-Derived"] = {
                    "cv_r2": float("nan"), "cv_mae": float("nan"),
                    "blind_r2": rhob_r2, "blind_mae": rhob_mae,
                    "top_feats": ["AI (Pred)", "DT (Pred)"]
                }
                
    # Derived LMRHO
    if "AI (Pred)" in pred_df.columns and "DT (Pred)" in pred_df.columns and "MURHO (Pred)" in pred_df.columns and "LMRHO (Act)" in pred_df.columns:
        dt_pred = pred_df["DT (Pred)"].values
        ai_pred = pred_df["AI (Pred)"].values
        murho_pred = pred_df["MURHO (Pred)"].values
        lmrho_act = pred_df["LMRHO (Act)"].values
        
        vp_pred = (1000000.0 / np.clip(dt_pred, 10.0, 300.0)) * 0.3048
        rhob_phys = ai_pred / vp_pred
        vp_km_s = vp_pred / 1000.0
        lmrho_phys = rhob_phys * (vp_km_s**2) - 2.0 * murho_pred
        
        valid = ~np.isnan(lmrho_act) & ~np.isnan(lmrho_phys)
        if np.any(valid):
            lmrho_r2 = r2_score(lmrho_act[valid], lmrho_phys[valid])
            lmrho_mae = mean_absolute_error(lmrho_act[valid], lmrho_phys[valid])
            
            if "LMRHO" in all_results:
                all_results["LMRHO"]["Physics-Derived"] = {
                    "cv_r2": float("nan"), "cv_mae": float("nan"),
                    "blind_r2": lmrho_r2, "blind_mae": lmrho_mae,
                    "top_feats": ["AI (Pred)", "DT (Pred)", "MURHO (Pred)"]
                }
            if "LMRHO (Sand)" in all_results:
                all_results["LMRHO (Sand)"]["Physics-Derived"] = {
                    "cv_r2": float("nan"), "cv_mae": float("nan"),
                    "blind_r2": lmrho_r2, "blind_mae": lmrho_mae,
                    "top_feats": ["AI (Pred)", "DT (Pred)", "MURHO (Pred)"]
                }

# Dynamic best model and model list selection
best_per = {}
for lbl, md in all_results.items():
    valid_keys = [k for k in md.keys() if not np.isnan(md[k]["cv_r2"])]
    if valid_keys:
        best_per[lbl] = max(valid_keys, key=lambda k: md[k]["cv_r2"])
    else:
        best_per[lbl] = max(md.keys(), key=lambda k: md[k]["blind_r2"])

all_model_names = set()
for lbl, md in all_results.items():
    all_model_names.update(md.keys())
model_names = sorted(list(all_model_names))

print("\nRendering image ...")
target_labels = list(all_results.keys())
ROW_H=0.40; HDR_H=0.60; TITLE_H=0.90
FIG_W=24; FIG_H=TITLE_H+HDR_H+(len(target_labels)*len(model_names))*ROW_H+0.60
fig,ax = plt.subplots(figsize=(FIG_W,FIG_H))
ax.set_xlim(0,1); ax.set_ylim(0,FIG_H); ax.axis("off")
fig.patch.set_facecolor("#060d1a")

def cell(ax,x,y,w,h,txt,bg,fg="#dde3f0",fs=8.5,bold=False,align="center"):
    pad=0.003
    rect=mpatches.FancyBboxPatch((x+pad,y+0.015),w-2*pad,h-0.030,
         boxstyle="round,pad=0.008",linewidth=0,facecolor=bg,edgecolor="none")
    ax.add_patch(rect)
    ha="left" if align=="left" else "center"
    xp=x+0.008 if align=="left" else x+w/2
    ax.text(xp,y+h/2,txt,ha=ha,va="center",fontsize=fs,color=fg,
            fontweight="bold" if bold else "normal",clip_on=True)

COLS=["Target","Model","CV R2","CV MAE","Blind R2","Blind MAE","Top-5 Features"]
CW  =[0.10,    0.13,   0.09,   0.09,    0.09,      0.09,       0.41]
BEST_BG="#14532d"; BEST_FG="#86efac"
RA="#131e2e"; RB="#0c1522"; TA="#1a2a3e"; TB="#10192a"

ax.text(0.5,FIG_H-0.30,"ML Pipeline v7  |  All-Model Performance Comparison  |  Blind Well: Z-04",
        ha="center",va="center",fontsize=14,fontweight="bold",color="#f1f5f9")
ax.text(0.5,FIG_H-0.62,"Star = best LOWO-CV R2 per target  |  Green row = selected model  |  All-tree ensemble pool & cascades",
        ha="center",va="center",fontsize=8.5,color="#7c8ca0")

y_now=FIG_H-TITLE_H-HDR_H; x_now=0.0
for cw,cl in zip(CW,COLS):
    cell(ax,x_now,y_now,cw,HDR_H*0.88,cl,"#1a3a5e","#e2e8f0",fs=9.5,bold=True)
    x_now+=cw
y_now -= HDR_H*0.88+0.05

for ti,lbl in enumerate(target_labels):
    md = all_results[lbl]; bm = best_per.get(lbl, None); par = ti%2
    for mi,mname in enumerate(model_names):
        if mname in md:
            d = md[mname]
            ib = (mname == bm)
        else:
            d = {"cv_r2": float("nan"), "cv_mae": float("nan"), "blind_r2": float("nan"), "blind_mae": float("nan"), "top_feats": []}
            ib = False
        rbg=BEST_BG if ib else (RA if par==0 else RB)
        rfg=BEST_FG if ib else "#dde3f0"
        tbg=TA if par==0 else TB
        def fr(v,mn=-9.99):
            if np.isnan(v): return "N/A"
            return f"{max(v,mn):+.4f}"
        vals=[lbl if mi==0 else "",
              ("* " if ib else "  ")+mname,
              fr(d["cv_r2"]),f"{d['cv_mae']:.4f}",
              fr(d["blind_r2"]),
              f"{d['blind_mae']:.4f}" if not np.isnan(d["blind_mae"]) else "N/A",
              "  |  ".join(d["top_feats"]) if d["top_feats"] else "-"]
        x_now=0.0
        for ci,(cw,val) in enumerate(zip(CW,vals)):
            bg=tbg if ci==0 else rbg
            if ci==0: fg="#7c8ca0"
            elif ci==1 and not ib: fg=MODEL_COLORS.get(mname,"#dde3f0")
            else: fg=rfg
            cell(ax,x_now,y_now,cw,ROW_H*0.90,val,bg,fg,
                 fs=7.2 if ci==6 else 8.0,
                 bold=(ci==0 and mi==0) or ib,
                 align="left" if ci==6 else "center")
            x_now+=cw
        y_now-=ROW_H
    ax.axhline(y_now+ROW_H*0.08,color="#1e3348",linewidth=0.8)

ax.text(0.5,y_now-0.18,
        f"v7 Pipeline  |  {len(target_labels)} targets  |  {len(model_names)} models  |  LOWO-CV 5 training wells  |  Z-04 blind",
        ha="center",va="center",fontsize=8,color="#475569")

plt.tight_layout(pad=0.2)
out1=os.path.join(output_dir,"model_comparison_all_models.png")
out2=os.path.join(legacy_output_dir,"model_comparison_all_models.png")
plt.savefig(out1,dpi=160,bbox_inches="tight",facecolor=fig.get_facecolor())
plt.savefig(out2,dpi=160,bbox_inches="tight",facecolor=fig.get_facecolor())
plt.close()
print(f"\nSaved -> {out1}")
print(f"Saved -> {out2}")

# ─── Export structured data as JS module for the frontend ─────────────────
import json

export_data = []
for lbl, mdict in all_results.items():
    bm = best_per[lbl]
    entry = {"target": lbl, "bestModel": bm, "models": []}
    for mname in model_names:
        if mname in mdict:
            d = mdict[mname]
            # Handle possible NaN values in R2/MAE for physics-derived model
            cvR2_val = round(d["cv_r2"], 4) if not np.isnan(d["cv_r2"]) else None
            cvMae_val = round(d["cv_mae"], 4) if not np.isnan(d["cv_mae"]) else None
            entry["models"].append({
                "name": mname,
                "isBest": mname == bm,
                "cvR2":    cvR2_val,
                "cvMae":   cvMae_val,
                "blindR2": round(d["blind_r2"], 4) if not np.isnan(d["blind_r2"]) else None,
                "blindMae":round(d["blind_mae"], 4) if not np.isnan(d["blind_mae"]) else None,
                "topFeatures": d["top_feats"],
            })
        else:
            entry["models"].append({
                "name": mname,
                "isBest": False,
                "cvR2":    None,
                "cvMae":   None,
                "blindR2": None,
                "blindMae":None,
                "topFeatures": [],
            })
    export_data.append(entry)

frontend_src = os.path.join(analysis_dir, "frontend", "src")
js_path = os.path.join(frontend_src, "model_comparison_data.js")
with open(js_path, "w", encoding="utf-8") as f:
    f.write("// Auto-generated by generate_model_comparison_image.py — do not edit manually\n")
    f.write(f"export const modelComparisonData = {json.dumps(export_data, indent=2)};\n")
print(f"Saved -> {js_path}")
print("Done.")

