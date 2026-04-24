import os
import pandas as pd
import numpy as np
import gc  # Added for memory management
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from google import genai
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from dotenv import load_dotenv

load_dotenv()

# ─── HYBRID ROUTER IMPORTS ───
from sklearn.linear_model import LogisticRegression as sk_LR
from sklearn.ensemble import RandomForestClassifier as sk_RF
from sklearn.tree import DecisionTreeClassifier as sk_DT

from glassboxml.models import LogisticRegression as gb_LR
from glassboxml.models import RandomForestClassifier as gb_RF
from glassboxml.models import DecisionTreeClassifier as gb_DT
from glassboxml.core import Momentum

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔐 SECURE API KEY INITIALIZATION
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

def compute_algorithmic_reweighing(df, sensitive_col, target_col):
    total_count = len(df)
    st_counts = df.groupby([sensitive_col, target_col]).size().reset_index(name='count_st')
    s_counts = df.groupby(sensitive_col).size().reset_index(name='count_s')
    t_counts = df.groupby(target_col).size().reset_index(name='count_t')
    
    df_weights = df[[sensitive_col, target_col]].copy()
    df_weights = df_weights.merge(st_counts, on=[sensitive_col, target_col], how='left')
    df_weights = df_weights.merge(s_counts, on=sensitive_col, how='left')
    df_weights = df_weights.merge(t_counts, on=target_col, how='left')
    
    p_s = df_weights['count_s'] / total_count
    p_t = df_weights['count_t'] / total_count
    p_s_and_t = df_weights['count_st'] / total_count
    
    weights = (p_s * p_t) / (p_s_and_t + 1e-9)
    return weights.fillna(1.0).values

def generate_offline_summary(model, dir_score, gap, status, mitigation, engine):
    dir_fmt = round(dir_score, 3)
    gap_fmt = round(gap, 1)

    if status == 'FAIR':
        if mitigation == 'reweighing':
            return f"The {model} model was audited via {engine} and successfully mitigated using Algorithmic Reweighing. By dynamically adjusting sample weights, the engine achieved a DIR of {dir_fmt} and closed the gap to {gap_fmt}%. It is mathematically certified as FAIR."
        else:
            return f"The baseline {model} model evaluated via {engine} passed the audit without mitigation. It achieved a DIR of {dir_fmt} with a minimal gap of {gap_fmt}%. The model is cleared for clinical deployment."
    else:
        return f"CRITICAL WARNING: The {model} model exhibits significant demographic bias under {engine}. The DIR is {dir_fmt}, resulting in an unacceptable gap of {gap_fmt}%. This model is categorized as BIASED."

def generate_gemini_summary(model, dir_score, gap, status, mitigation, engine):
    if not client:
        return generate_offline_summary(model, dir_score, gap, status, mitigation, engine)

    prompt = f"""
    AI fairness auditor summary. 3 sentences. analytical.
    {model} via {engine}. Mitigation: {mitigation}.
    DIR: {round(dir_score, 3)}, Gap: {round(gap, 1)}%, Status: {status}.
    """
    try:
        response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        return response.text
    except Exception as e:
        print(f"⚠️ API Fallback triggered: {str(e)}")
        return generate_offline_summary(model, dir_score, gap, status, mitigation, engine)

@app.route('/api/audit', methods=['POST'])
def run_audit():
    filepath = None
    try:
        print("\n🚀 [NodeBias v2.0] Engine Hit! Processing payload...")

        if 'dataset' not in request.files:
            return jsonify({'error': 'No dataset uploaded'}), 400

        file = request.files['dataset']
        model_type = request.form.get('modelType', 'Random Forest')
        target_col = request.form.get('targetColumn')
        sensitive_col = request.form.get('sensitiveColumn')
        mitigation = request.form.get('mitigation', 'none')

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # 1. LOAD & DOWNSAMPLE (SIGKILL PREVENTION)
        df = pd.read_csv(filepath)
        if len(df) > 7000:
            print(f"⚠️ Downsampling {len(df)} rows to 7000 for RAM stability.")
            df = df.sample(n=7000, random_state=42).reset_index(drop=True)

        df.replace(['?', 'NA', 'N/A', '', 'NULL'], np.nan, inplace=True)
        df = df.dropna(subset=[target_col, sensitive_col]).reset_index(drop=True)
        df.columns = df.columns.str.strip()

        if target_col not in df.columns or sensitive_col not in df.columns:
            return jsonify({'error': f'Columns {target_col} or {sensitive_col} missing.'}), 400

        # 2. Universal Binarization
        if len(df[target_col].unique()) > 2:
            if target_col.strip() == 'readmitted':
                df[target_col] = df[target_col].apply(lambda x: 0 if str(x).strip().upper() == 'NO' else 1)
            else:
                if pd.api.types.is_numeric_dtype(df[target_col]):
                    threshold = df[target_col].median()
                    df[target_col] = (df[target_col] > threshold).astype(int)
                else:
                    top_val = df[target_col].value_counts().index[0]
                    df[target_col] = (df[target_col] == top_val).astype(int)

        # 3. Feature Engineering
        le_dict = {}
        for col in df.columns:
            if col == target_col: continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                if col == sensitive_col: le_dict[col] = le
            else:
                df[col] = df[col].fillna(df[col].median())

        # 4. Mitigation & Splitting
        sample_weights = compute_algorithmic_reweighing(df, sensitive_col, target_col) if mitigation == 'reweighing' else None
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, sample_weights if sample_weights is not None else np.ones(len(y)),
            test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 5. Hybrid Training (Optimized Estimators for RAM)
        if mitigation == 'reweighing':
            engine_used = "Sklearn"
            if model_type == 'Logistic Regression': model = sk_LR(max_iter=1000)
            elif model_type == 'Decision Tree': model = sk_DT(max_depth=6)
            else: model = sk_RF(n_estimators=5, max_depth=6)
            model.fit(X_train_scaled, y_train, sample_weight=w_train)
            predictions = model.predict(X_test_scaled)
        else:
            engine_used = "GlassBoxML"
            if model_type == 'Logistic Regression': model = gb_LR(optimizer=Momentum(learning_rate=0.01))
            elif model_type == 'Decision Tree': model = gb_DT(max_depth=6)
            else: model = gb_RF(n_trees=5, max_depth=6)
            model.fit(X_train_scaled, y_train.values)
            predictions = model.predict(X_test_scaled)

        # 6. Fairness Audit
        df_audit = X_test.copy()
        df_audit['Predicted'] = predictions
        group_rates = {}
        for g in df_audit[sensitive_col].unique():
            group_df = df_audit[df_audit[sensitive_col] == g]
            if len(group_df) < 5: continue
            favorable_rate = float(group_df['Predicted'].mean())
            name = le_dict[sensitive_col].inverse_transform([int(g)])[0] if sensitive_col in le_dict else str(g)
            group_rates[name] = favorable_rate

        rates = list(group_rates.values())
        dir_score = min(rates)/max(rates) if rates and max(rates) > 0 else 1.0
        gap = (max(rates) - min(rates)) * 100
        status = "FAIR" if dir_score >= 0.8 else "BIASED"
        summary = generate_gemini_summary(model_type, dir_score, gap, status, mitigation, engine_used)

        response_data = {
            'dir_after': round(dir_score, 3),
            'disparity_gap_after': round(gap, 2),
            'group_rates_after': group_rates,
            'features_used': len(X.columns),
            'status': status,
            'strategy_used': 'Reweighing' if mitigation == 'reweighing' else 'Baseline',
            'ai_summary': summary
        }

        # 7. AGGRESSIVE CLEANUP BEFORE RESPONSE
        del df, X, y, X_train, X_test, X_train_scaled, X_test_scaled, df_audit
        gc.collect()

        return jsonify(response_data)

    except Exception as e:
        print(f"🔥 Engine Error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
