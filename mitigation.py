import pandas as pd
import numpy as np

def compute_sample_weight(df, sensitive_col, target_col):
    """
    Calculates statistical reweighing to balance historical dataset bias.
    Returns an array of weights mapping exactly to the DataFrame index.
    """
    n_total = len(df)
    weights = pd.Series(index=df.index, dtype=float)

    groups = df[sensitive_col].unique()
    classes = df[target_col].unique()

    for g in groups:
        for c in classes:
            # Expected Probability: P(Group) * P(Class)
            p_g = len(df[df[sensitive_col] == g]) / n_total
            p_c = len(df[df[target_col] == c]) / n_total

            # Observed Probability: P(Group AND Class)
            mask = (df[sensitive_col] == g) & (df[target_col] == c)
            p_g_c = len(df[mask]) / n_total

            # Calculate the balancing weight
            weight = (p_g * p_c) / p_g_c if p_g_c > 0 else 1.0

            # Apply weight to all matching records
            weights[mask] = weight

    return weights.values
