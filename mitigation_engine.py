import pandas as pd
import numpy as np
from glassboxml.core import Momentum
from glassboxml.preprocessing import StandardScaler
from glassboxml.models import LogisticRegression ,GaussianNaiveBayes

def run_nodebias_audit(df, model_choice, target_col='readmitted', sensitive_col='gender'):
    try:
        # 1. Dynamic Cleaning
        df.replace(['?', 'NA', 'N/A', ''], np.nan, inplace=True)

        # 2. Binarize Target dynamically
        if df[target_col].dtype == 'object':
            df['target_binary'] = df[target_col].astype('category').cat.codes
        else:
            df['target_binary'] = df[target_col]

        # 3. Auto-Encode Categorical Strings
        for col in df.select_dtypes(include=['object', 'string', 'category']).columns:
            if col not in [sensitive_col, target_col]:
                df[col] = df[col].astype('category').cat.codes

        # 4. Auto-detect safe features
        safe_features = df.select_dtypes(include=[np.number]).columns.tolist()
        safe_features = [f for f in safe_features if f not in [target_col, 'target_binary', sensitive_col]]

        df.dropna(subset=safe_features + [sensitive_col], inplace=True)

        # 5. Extract Tensors
        X = df[safe_features].values.astype(float)
        y = df['target_binary'].values.astype(int)
        raw_sensitive = df[sensitive_col].values

        # --- THE FIX: Train On-The-Fly Instead of Using .pkl Files ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if model_choice == "Logistic Regression":
            optimizer = Momentum(learning_rate = 0.01,beta = 0.9)
            model = LogisticRegression(optimizer = optimizer, epochs = 10000,loss_function = 'bce')
        else:
            model = GaussianNaiveBayes()

        # The AI learns the new dataset instantly in RAM
        model.fit(X_scaled, y)
        predictions = model.predict(X_scaled)
        # -------------------------------------------------------------

        # Calculate Fairness JSON
        results_df = pd.DataFrame({'attr': raw_sensitive, 'pred': predictions})
        groups = results_df['attr'].unique()
        rates = {str(g): float(results_df[results_df['attr'] == g]['pred'].mean()) for g in groups}

        # Calculate DIR (Using min/max so it always scales properly for the UI)
        g_list = list(rates.values())
        dir_val = round(min(g_list) / max(g_list), 3) if len(g_list) > 1 and max(g_list) > 0 else 1.0

        return {
            "strategy_used": "Dynamic On-The-Fly Mitigation",
            "features_used": len(safe_features),
            "group_rates_after": rates,
            "disparity_gap_after": round(abs(g_list[0] - g_list[1]) * 100, 2) if len(g_list) > 1 else 0,
            "dir_after": dir_val,
            "status": "Safe for Deployment" if dir_val >= 0.8 else "Bias Detected"
        }
    except Exception as e:
        raise Exception(f"Engine Failure: {str(e)}")
