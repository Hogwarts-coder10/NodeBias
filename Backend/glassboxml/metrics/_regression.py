import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Calculates the average squared difference between predictions and reality.
    Lower is better (0.0 is perfect).
    """
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    """
    Calculates the R-squared (Coefficient of Determination).
    Best possible score is 1.0. A score of 0.0 means the model is no better 
    than just guessing the average of the data.
    """
    # Sum of Squared Residuals (The model's mistakes)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Total Sum of Squares (The variance of the raw data)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Prevent division by zero if the data is totally flat
    if ss_tot == 0:
        return 0.0
        
    return 1 - (ss_res / ss_tot)