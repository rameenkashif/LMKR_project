import os
import re
import json
import pandas as pd

# Load well name from train_model.py
train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_training", "train_model.py"))
with open(train_path, "r", encoding="utf-8") as f:
    code = f.read()

match = re.search(r'BLIND_WELL\s*=\s*["\']([^"\']+)["\']', code)
well_name = match.group(1) if match else "Unknown"

# Load comparison CSV
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_outputs_v7", "blind_well_actual_vs_predicted.csv"))
df = pd.read_csv(csv_path)
data = df.to_dict(orient="records")

# Write to data.js
data_js_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src", "data.js"))
out_content = f'export const blindWellName = "{well_name}";\nexport const blindData = {json.dumps(data, indent=2)};\n'

with open(data_js_path, "w", encoding="utf-8") as f:
    f.write(out_content)

print(f"Successfully synchronized {well_name} data to frontend/src/data.js")
