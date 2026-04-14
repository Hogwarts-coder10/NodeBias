import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from google import genai

# Custom Engine Imports
from data_pipeline import clean_medical_data
from glassboxml.models import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier
from glassboxml.core import Momentum
from glassboxml.preprocessing import StandardScaler
from mitigation import compute_sample_weight
from scanner import universal_sensitive_scanner

app = Flask(__name__)

# Ensure a temporary directory exists for the uploaded datasets
UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/api/audit", methods=["POST"])
def audit_model():
    print("\n🌳 Python Backend Hit! Initializing GlassBoxML...")

    # 1. Parse incoming configuration
    model_type = request.form.get("modelType", "Random Forest")
    target_col = request.form.get("targetColumn", "readmitted")
    sensitive_col = request.form.get("sensitiveColumn", "gender")

    # 2. Handle the uploaded CSV
    if "dataset" not in request.files:
        return jsonify({"error": "No dataset uploaded"}), 400

    file = request.files["dataset"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        # 3. Load Data
        df = pd.read_csv(file_path)

        if target_col == "readmitted" and "<30" in df[target_col].values:
            df[target_col] = df[target_col].apply(
                lambda x: 1 if x in ["<30", ">30"] else 0
            )

        # 4. Universal Pipeline
        X, y, raw_sensitive, features, _ = clean_medical_data(
            df, target_col=target_col, sensitive_col=sensitive_col, is_mitigated=False
        )

        # 5. Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 6. The Hybrid Mitigation Router
        mitigation_strategy = request.form.get('mitigation', 'none')

        if mitigation_strategy == 'reweighing':
            print("⚖️ Applying Algorithmic Reweighing (Routing to Sklearn Engine)...")
            sample_weights = compute_sample_weight(df, sensitive_col, target_col)

            # 🚀 Use sklearn for the fix to guarantee mathematical weight processing
            if model_type == 'Logistic Regression':
                from sklearn.linear_model import LogisticRegression as SklearnLR
                model = SklearnLR(max_iter=1000)
                engine_name = "Sklearn Logistic Regression (Active Mitigation)"
            elif model_type == 'Decision Tree':
                from sklearn.tree import DecisionTreeClassifier as SklearnDT
                model = SklearnDT(max_depth=4)
                engine_name = "Sklearn Decision Tree (Active Mitigation)"
            else:
                from sklearn.ensemble import RandomForestClassifier as SklearnRF
                model = SklearnRF(n_estimators=10, max_depth=14)
                engine_name = "Sklearn Random Forest (Active Mitigation)"

            # 🚨 Train WITH fairness weights passed into the engine!
            model.fit(X_scaled, y, sample_weight=sample_weights)

        else:
            print("▶️ Running Baseline (Routing to Custom GlassBoxML Engine)...")

            # 🏆 Use your custom framework to show off the from-scratch engineering
            if model_type == 'Logistic Regression':
                optimizer = Momentum(learning_rate=0.01, beta=0.9)
                model = LogisticRegression(optimizer=optimizer, epochs=1000)
            elif model_type == 'Decision Tree':
                model = DecisionTreeClassifier(max_depth=14)
            else:
                model = RandomForestClassifier(n_trees=10, max_depth=14)

            engine_name = f"GlassBoxML {model_type} (Baseline)"

            # Train WITHOUT weights (since the custom engine doesn't support them yet)
            model.fit(X_scaled, y)

        predictions = model.predict(X_scaled)

        # 7. Generate Bias Report
        results_df = pd.DataFrame({"attr": raw_sensitive, "pred": predictions})
        rates = {
            str(g): float(results_df[results_df["attr"] == g]["pred"].mean())
            for g in results_df["attr"].unique()
        }

        g_values = [v for v in rates.values() if v > 0]
        dir_val = round(min(g_values) / max(g_values), 3) if len(g_values) > 1 else 1.0
        gap = (
            round(abs(max(g_values) - min(g_values)) * 100, 2)
            if len(g_values) > 1 else 0
        )
        status_pass = dir_val >= 0.8

        print(f"🌳 Audit Complete. DIR: {dir_val}")

        # 8. Google Gemini Integration (Updated SDK)
        prompt = f"""
Act as an AI Fairness Auditor. Review the following ML model audit:
- Model: {engine_name}
- Sensitive Attribute: {sensitive_col}
- Disparate Impact Ratio (DIR): {dir_val} (Passing threshold is 0.80)
- Mitigation Strategy: {'Active Reweighing' if mitigation_strategy == 'reweighing' else 'None'}

Write a concise, 2-sentence executive summary of these results for a non-technical stakeholder.
State whether the model passed or failed, and briefly explain why based on the DIR.
"""
        try:
            client = genai.Client(api_key = "YOUR_API_KEY")
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt.strip()
            )
            ai_summary = response.text.strip()
        except Exception as e:
            ai_summary = "AI Summary unavailable. Please check API key or network."
            print(f"Gemini Error: {e}")

        # 9. Package JSON
        response_data = {
            "strategy_used": "Active Reweighing" if mitigation_strategy == 'reweighing' else "Baseline",
            "model_used": engine_name,
            "features_used": X.shape[1] if hasattr(X, "shape") else 42,
            "group_rates_after": {
                "Female": rates.get("Female", 0.0),
                "Male": rates.get("Male", 0.0),
            },
            "disparity_gap_after": gap,
            "dir_after": dir_val,
            "status": "Safe for Deployment" if status_pass else "Bias Detected",
            "ai_summary": ai_summary
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route("/api/scan-dataset", methods=["POST"])
def scan_dataset():
    if "dataset" not in request.files:
        return jsonify({"error": "No dataset uploaded"}), 400

    file = request.files["dataset"]

    try:
        df_sample = pd.read_csv(file, nrows=50)
        detected = universal_sensitive_scanner(df_sample)

        return jsonify({
            "status": "success",
            "detected_sensitive_columns": [item["column"] for item in detected],
            "details": detected,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 GlassBoxML API Engine starting on port 5000...")
    app.run(port=5000, debug=True)
