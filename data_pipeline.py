import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def clean_medical_data(df, target_col='is_readmitted', sensitive_col='gender', is_mitigated=False):
    """
    Fully dynamic NodeBias Data Pipeline.
    Auto-encodes text, detects features, and manages AI training shadows.
    """
    logging.info(f"Initializing Dynamic Pipeline (Target: {target_col}, Sensitive: {sensitive_col})")

    # 1. Standardize Nulls globally
    df.replace(['?', 'NA', 'N/A', ''], np.nan, inplace=True)

    # 2. Schema Validation
    if target_col not in df.columns or sensitive_col not in df.columns:
        raise ValueError(f"Dataset must contain the target '{target_col}' and sensitive '{sensitive_col}'.")

    # Drop rows missing the critical audit targets
    df.dropna(subset=[target_col, sensitive_col], inplace=True)

    # 3. Auto-Encode Categorical Strings (Do this FIRST)
    for col in df.select_dtypes(include=['object', 'string', 'category']).columns:
        if col not in [sensitive_col, target_col]:
            df[col] = df[col].astype('category').cat.codes

    # 4. Auto-detect all safe numerical features (Defines safe_features!)
    numeric_cols = df.select_dtypes(include=[np.number, 'float64', 'int64']).columns.tolist()
    safe_features = [col for col in numeric_cols if col not in [target_col, sensitive_col]]

    if len(safe_features) == 0:
         raise ValueError("No valid numerical features found in dataset to train the model.")

    # 5. Create a "Shadow" numeric column for the AI to train on
    df['sensitive_numeric'] = df[sensitive_col].astype('category').cat.codes

    # 6. Routing logic based on pipeline phase
    if is_mitigated:
        # Fairness Through Unawareness: Hide both the text and numeric sensitive columns
        features_to_use = safe_features
        df.dropna(subset=features_to_use + [sensitive_col], inplace=True)
    else:
        # Baseline AI: Feed it the NUMERIC shadow column so the math doesn't crash
        features_to_use = safe_features + ['sensitive_numeric']
        df.dropna(subset=features_to_use + [sensitive_col], inplace=True)

    # 7. Extract Tensors for the custom GlassBoxML engines
    X = df[features_to_use].values.astype(float)
    y = df[target_col].values.astype(float)

    # Grab the RAW text column for the JSON report so the UI looks great
    raw_sensitive_attributes = df[sensitive_col].values

    logging.info(f"Pipeline Complete. Auto-detected {len(safe_features)} features. Processed {len(df)} records.")

    return X, y, raw_sensitive_attributes, features_to_use, df
