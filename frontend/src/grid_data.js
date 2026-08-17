// Stub: ml_training/generate_grid_predictions.py still targets the retired
// ml_outputs_v8/ model format (separate *_scaler.joblib / *_selected_indices.joblib
// files) and was never updated for the V11 model save format
// (*_model.joblib + *_feature_indices.joblib, see precompute_v11_slice_predictions.py).
// Until that script is ported to V11, this stub keeps the frontend build green;
// GridPredictorTab and Volume3dTab already null-check gridData.
export const gridData = null;
