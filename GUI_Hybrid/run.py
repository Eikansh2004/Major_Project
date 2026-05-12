import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from custom_layers import build_hu_model, build_disease_model, load_model_weights

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HU_MODEL_PATH = os.path.join(BASE_DIR, "hybrid_cnn_vit_hu_best.h5")
DISEASE_MODEL_PATH = os.path.join(BASE_DIR, "hybrid_cnn_vit_disease_best.h5")
HU_DATASET_PATH = os.path.join(BASE_DIR, "healthy_unhealthy1.csv")

# ---------------------------------------------------------------------------
# Precompute z-score stats from the HU training dataset (CP1 needs this)
# The HU model was trained with input_dim=1025, meaning all 1025 features
# from the CSV (excluding the label column) were used.
# ---------------------------------------------------------------------------
print("Loading normalization statistics from healthy_unhealthy1.csv ...")
_hu_data = pd.read_csv(HU_DATASET_PATH)
X_ALL = _hu_data.iloc[:, :-1].values          # (N, 1025) — all columns except label
HU_INPUT_DIM = X_ALL.shape[1]                 # 1025
X_MEAN = X_ALL.mean(axis=0)                   # (1025,)
X_STD = X_ALL.std(axis=0) + 1e-8              # (1025,)
del _hu_data, X_ALL
print(f"Normalization stats ready (input_dim={HU_INPUT_DIM}).")

# ---------------------------------------------------------------------------
# Load models once at startup (rebuild architecture + load weights)
# ---------------------------------------------------------------------------
print("Building and loading Hybrid CNN-ViT models ...")
hu_model = load_model_weights(build_hu_model, HU_MODEL_PATH, input_dim=HU_INPUT_DIM)
disease_model = load_model_weights(build_disease_model, DISEASE_MODEL_PATH)
print("Models loaded successfully.")

# Disease labels — same order as training one-hot encoding
# CP2 notebook: {0: narco, 1: ins, 2: nfle, 3: plm, 4: rbd}
DISEASE_LABELS = ['NARCO', 'INS', 'NFLE', 'PLM', 'RBD']
DISEASE_FULL_NAMES = {
    'INS': 'Insomnia',
    'NFLE': 'Nocturnal Frontal-Lobe Epilepsy',
    'NARCO': 'Narcolepsy',
    'RBD': 'REM Sleep Behaviour Disorder',
    'PLM': 'Periodic Leg Movement',
}

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def classify():
    result = {}

    if request.method == "POST":
        uploaded = request.files.get("myfile")
        if uploaded and uploaded.filename.endswith(".csv"):
            # Read raw feature vector from CSV
            raw = np.loadtxt(uploaded, delimiter=",").flatten()

            if raw.size < 1024:
                result["error"] = (
                    f"CSV has {raw.size} values, need at least 1024."
                )
                return render_template("form.html", result=result)

            # Pad to HU_INPUT_DIM (1025) if only 1024 EEG features were provided.
            # The 1025th column in the training data was the CAP phase indicator;
            # we pad with 0 (phase B) as a safe default when it is absent.
            if raw.size < HU_INPUT_DIM:
                raw = np.concatenate([raw, np.zeros(HU_INPUT_DIM - raw.size)])

            # --- CP1: Healthy / Unhealthy (z-score normalized) ---
            x_hu = raw[:HU_INPUT_DIM].reshape(1, HU_INPUT_DIM)
            x_normalized = (x_hu - X_MEAN) / X_STD
            hu_pred = hu_model.predict(x_normalized, verbose=0)
            hu_class = int(np.argmax(hu_pred, axis=1)[0])  # 0=Healthy, 1=Unhealthy

            if hu_class == 1:
                result["health"] = "Positive (Unhealthy)"

                # --- CP2: Disease Classification (raw features, 1024 dims) ---
                x_disease = raw[:1024].reshape(1, 1024)
                disease_pred = disease_model.predict(x_disease, verbose=0)
                disease_idx = int(np.argmax(disease_pred, axis=1)[0])
                disease_code = DISEASE_LABELS[disease_idx]
                result["disease"] = (
                    f"{disease_code} — {DISEASE_FULL_NAMES[disease_code]}"
                )
            else:
                result["health"] = "Negative (Healthy)"
        else:
            result["error"] = "Please upload a valid .csv file."

    return render_template("form.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, port=3232)
