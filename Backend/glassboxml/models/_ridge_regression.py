import numpy as np
from glassboxml.core._base_model import GlassBoxModel

class RidgeRegression(GlassBoxModel):
    """
    Transparent Ridge Regression (L2 Regularization).
    Prevents overfitting and handles multicollinearity by penalizing large weights.
    """
    def __init__(self, optimizer, alpha=1.0, epochs=1000):
        super().__init__()
        self.optimizer = optimizer
        self.alpha = alpha  # The regularization strength
        self.epochs = epochs

    def check_assumptions(self, X, y):
        self.failure_modes = []
        variances = np.var(X, axis=0)
        max_var = np.max(variances)
        min_var = np.min(variances)
        
        if min_var > 0 and (max_var / min_var) > 10:
            self.failure_modes.append(
                f"Unscaled features detected (Max Var: {max_var:.2f}, Min Var: {min_var:.2f}). "
                "Ridge Regression is highly sensitive to feature scales. "
                "Features with larger scales will be penalized less. Please standardize your X data."
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
            
            # 2. Compute Loss (MSE + L2)
            mse_loss = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
            l2_penalty = (self.alpha / 2) * np.sum(self.coef_ ** 2)
            loss = mse_loss + l2_penalty
            
            # 3. Compute Gradients (MSE + L2 Derivative)
            grads = {
                'dw': (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (self.alpha * self.coef_),
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

    def predict(self, X):
        return self.decision_function(X)

    def explain(self):
        if not self.is_fitted:
            return "Model is not fitted."
            
        equation = "y = "
        terms = [f"({w:.4f} * x{i+1})" for i, w in enumerate(self.coef_)]
        equation += " + ".join(terms) + f" + {self.intercept_:.4f}"
        
        return (
            "--- GlassBox Explanation: Ridge Regression (L2) ---\n"
            f"Equation: {equation}\n"
            f"Regularization Strength (Alpha): {self.alpha}\n"
            "Interpretation: Ridge shrinks all feature weights to prevent extreme reliance on any "
            "single feature, ensuring a smoother, more robust model without eliminating features entirely."
        )