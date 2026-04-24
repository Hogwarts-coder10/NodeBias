import numpy as np
from glassboxml.core._base_model import GlassBoxModel

class LassoRegression(GlassBoxModel):
    """
    Transparent Lasso Regression (L1 Regularization).
    Performs automatic feature selection by forcing useless feature weights to zero.
    """
    def __init__(self, optimizer, alpha=1.0, epochs=1000):
        super().__init__()
        self.optimizer = optimizer
        self.alpha = alpha  
        self.epochs = epochs

    def check_assumptions(self, X, y):
        self.failure_modes = []
        variances = np.var(X, axis=0)
        max_var, min_var = np.max(variances), np.min(variances)
        
        if min_var > 0 and (max_var / min_var) > 10:
            self.failure_modes.append(
                "Unscaled features detected. Lasso Regression will aggressively "
                "delete features with smaller scales. Please standardize X."
            )
        return self.failure_modes

    def fit(self, X, y):
        self._store_dataset_stats(X)
        n_samples, n_features = X.shape
        
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        
        self.check_assumptions(X, y)

        for epoch in range(self.epochs):
            # 1. Forward Pass
            y_pred = np.dot(X, self.coef_) + self.intercept_
            
            # 2. Compute Loss (MSE + L1)
            mse_loss = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
            l1_penalty = self.alpha * np.sum(np.abs(self.coef_))
            loss = mse_loss + l1_penalty
            
            # 3. Compute Gradients (MSE + L1 Derivative)
            grads = {
                'dw': (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (self.alpha * np.sign(self.coef_)),
                'db': (1 / n_samples) * np.sum(y_pred - y)
            }
            
            # 4. Clean One-Liner Optimizer Update
            self.coef_, self.intercept_ = self.optimizer.update(
                self.coef_, self.intercept_, grads['dw'], grads['db']
            )
            
            # 5. Record Step
            self._record_step(epoch_loss=loss, epoch_gradients={'dw': grads['dw'].copy(), 'db': grads['db']})
            
        self.is_fitted = True
        self.training_error = self.loss_history[-1]
        
        # GlassBox Feature Selection: Snap tiny weights to exact 0
        tolerance = 1e-4
        zeroed_out = np.sum(np.abs(self.coef_) < tolerance)
        self.coef_[np.abs(self.coef_) < tolerance] = 0.0
        
        if zeroed_out > 0:
            self.failure_modes.append(
                f"Feature Selection: Lasso forced {zeroed_out} out of {n_features} features to exactly 0.0."
            )

    def predict(self, X):
        return self.decision_function(X)

    def explain(self):
        if not self.is_fitted:
            return "Model is not fitted."
            
        zeroed_out = np.sum(self.coef_ == 0.0)
        total_features = len(self.coef_)
        
        equation = "y = "
        terms = [f"({w:.4f} * x{i+1})" for i, w in enumerate(self.coef_) if w != 0.0]
        
        if terms:
            equation += " + ".join(terms) + f" + {self.intercept_:.4f}"
        else:
            equation += f"{self.intercept_:.4f} (All features eliminated)"
            
        return (
            "--- GlassBox Explanation: Lasso Regression (L1) ---\n"
            f"Active Equation: {equation}\n"
            f"Feature Selection: {zeroed_out} out of {total_features} features were completely eliminated (weight = 0).\n"
            "Interpretation: Lasso acts as an automatic feature selector, deleting non-predictive variables "
            "to leave you with the simplest possible model."
        )