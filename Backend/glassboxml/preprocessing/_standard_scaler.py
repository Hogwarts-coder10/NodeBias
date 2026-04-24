import numpy as np

class StandardScaler:
    """Standardizes features by removing the mean and scaling to unit variance."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.is_fitted = False

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0  # Prevent ZeroDivisionError
        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("GlassBox Error: Scaler has not been fitted yet. Call .fit(X) first.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def explain(self):
        if not self.is_fitted:
            return "Scaler is not fitted."
        return (
            "--- GlassBox Explanation: StandardScaler ---\n"
            f"Feature Means Shifted: {np.round(self.mean_, 4)}\n"
            f"Feature Scales (Std Dev): {np.round(self.scale_, 4)}\n"
            "Interpretation: Features have been centered at 0 and scaled to have a variance of 1. "
            "This ensures distance metrics and gradients treat all columns equally."
        )
    

class PolynomialFeatures:
    """Generates polynomial and interaction features."""
    def __init__(self, degree=2):
        self.degree = degree

    def transform(self, X):
        n_samples, n_features = X.shape
        
        if self.degree > 5:
            print(f"⚠️ CAPACITY WARNING: Degree {self.degree} polynomial requested. "
                  "Expect extreme variance (Runge's phenomenon) and severe overfitting.")
        if self.degree >= n_samples:
            print(f"⚠️ FATAL WARNING: Degree ({self.degree}) >= Number of samples ({n_samples}). "
                  "The model will perfectly memorize the noise.")

        X_poly = np.copy(X)
        for d in range(2, self.degree + 1):
            X_poly = np.hstack((X_poly, X ** d))
        return X_poly

    def fit_transform(self, X):
        return self.transform(X)