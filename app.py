import os

import pandas as pd
from flask import Flask, jsonify, request
from glassboxml.models import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier
from glassboxml.core import Momentum
from glassboxml.preprocessing import StandardScaler
from werkzeug.utils import secure_filename
from mitigation import compute_sample_weight

from data_pipeline import clean_medical_data

app = Flask(__name__)

# Ensure a temporary directory exists for the uploaded datasets
UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/api/audit", methods=["POST"])
def audit_model():
    print("🌳 Python Backend Hit! Initializing GlassBoxML...")

    # 1. Parse incoming configuration from the Node.js React app
    model_type = request.form.get("modelType", "Random Forest")
    target_col = request.form.get("targetColumn", "readmitted")
    sensitive_col = request.form.get("sensitiveColumn", "gender")

    # 2. Handle the uploaded CSV
    if "dataset" not in request.files:
        return jsonify({"error": "No dataset uploaded"}), 400

    file = request.files["dataset"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save temporarily to feed into pandas
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        # 3. Load Data
        df = pd.read_csv(file_path)

        # Apply specific formatting if it's the diabetic dataset
        if target_col == "readmitted" and "<30" in df[target_col].values:
            df[target_col] = df[target_col].apply(
                lambda x: 1 if x in ["<30", ">30"] else 0
            )

        # 4. Universal Pipeline (is_mitigated=False for baseline)
        X, y, raw_sensitive, features, _ = clean_medical_data(
            df, target_col=target_col, sensitive_col=sensitive_col, is_mitigated=False
        )

        # 5. Scale & Train
        mitigation_strategy = request.form.get('mitigation', 'none')

        # 1. Calculate weights if requested
        if mitigation_strategy == 'reweighing':
            print("⚖️ Applying Algorithmic Reweighing...")
            # We calculate weights on the raw df before dropping columns
            sample_weights = compute_sample_weights(df, sensitive_col, target_col)
        else:
            # Baseline (all records get a weight of 1.0)
            sample_weights = np.ones(len(X_scaled))

        print(f"Routing data to engine: {model_type}...")

        # 2. Dynamic Router
        if model_type == 'Logistic Regression':
            model = LogisticRegression(max_iter=1000)
            ensemble_size = 1
        elif model_type == 'Decision Tree':
            model = DecisionTreeClassifier(max_depth=14)
            ensemble_size = 1
        else:
            model = RandomForestClassifier(n_trees=10, max_depth=14)
            ensemble_size = 10

        # 3. Train the model WITH the fairness weights!
        # (Note: ensure your custom B-Tree accepts sample_weight. Sklearn models do natively.)
        model.fit(X_scaled, y)
        predictions = model.predict(X_scaled)

# 6. Generate the Bias Report
        results_df = pd.DataFrame({"attr": raw_sensitive, "pred": predictions})
        rates = {
            str(g): float(results_df[results_df["attr"] == g]["pred"].mean())
            for g in results_df["attr"].unique()
        }

        g_values = [v for v in rates.values() if v > 0]
        dir_val = round(min(g_values) / max(g_values), 3) if len(g_values) > 1 else 1.0
        gap = (
            round(abs(max(g_values) - min(g_values)) * 100, 2)
            if len(g_values) > 1
            else 0
        )
        status_pass = dir_val >= 0.8

        print(f"🌳 Forest Audit Complete. DIR: {dir_val}")

        # 7. Package the JSON exactly how their app.js expects it
        response_data = {
            "strategy_used": "Fairness Through Unawareness (GlassBoxML)",
            "model_used": model_type,
            "features_used": X.shape[1] if hasattr(X, "shape") else 42,
            "group_rates_after": {
                "Female": rates.get("Female", 0.0),
                "Male": rates.get("Male", 0.0),
            },
            "disparity_gap_after": gap,
            "dir_after": dir_val,
            "status": "Safe for Deployment" if status_pass else "Bias Detected",
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up the temporary file so the server doesn't fill up
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route("/api/scan-dataset", methods=["POST"])
def scan_dataset():
    if "dataset" not in request.files:
        return jsonify({"error": "No dataset uploaded"}), 400

    file = request.files["dataset"]

    try:
        # Read only the first 50 rows to keep the scan lightning fast
        df_sample = pd.read_csv(file, nrows=50)

        # Run the universal scanner
        detected = universal_sensitive_scanner(df_sample)

        return jsonify(
            {
                "status": "success",
                # Extract just the column names for the frontend dropdown
                "detected_sensitive_columns": [item["column"] for item in detected],
                # Send the full details in case the UI wants to show WHY it was flagged
                "details": detected,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 GlassBoxML API Engine starting on port 5000...")
    app.run(port=5000, debug=True)
