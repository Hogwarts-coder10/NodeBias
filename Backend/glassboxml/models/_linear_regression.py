import numpy as np
from glassboxml.core._base_model import GlassBoxModel

class LinearRegression(GlassBoxModel):
    """Transparent Linear Regression."""
    
    def __init__(self, optimizer, epochs=1000):
        super().__init__()
        self.optimizer = optimizer
        self.epochs = epochs

    def check_assumptions(self, X, y):
        self.failure_modes = []
        correlation_matrix = np.corrcoef(X, rowvar=False)
        if X.shape[1] > 1:
            upper_tri = np.triu(np.abs(correlation_matrix), k=1)
            if np.any(upper_tri > 0.95):
                self.failure_modes.append(
                    "High multicollinearity detected. Gradients may oscillate."
                )
        return self.failure_modes

    def fit(self, X, y):
        self._store_dataset_stats(X)
        n_samples, n_features = X.shape
        
        # Initialize standard attributes
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        
        self.check_assumptions(X, y)

        for epoch in range(self.epochs):
            # 1. Forward Pass
            y_pred = np.dot(X, self.coef_) + self.intercept_
            
            # 2. Compute Loss & Gradients
            loss = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
            grads = {
                'dw': (1 / n_samples) * np.dot(X.T, (y_pred - y)),
                'db': (1 / n_samples) * np.sum(y_pred - y)
            }
            
            # 3. Clean One-Liner Optimizer Update
            self.coef_, self.intercept_ = self.optimizer.update(
                self.coef_, self.intercept_, grads['dw'], grads['db']
            )
            
            # 4. Record Step
            self._record_step(epoch_loss=loss, epoch_gradients={'dw': grads['dw'].copy(), 'db': grads['db']})
            
        self.is_fitted = True
        self.training_error = self.loss_history[-1]

    def predict(self, X):
        return self.decision_function(X)

    def explain(self):
        """Translates the learned weights into a readable equation."""
        if not self.is_fitted:
            return "Model is not fitted."
            
        equation = "y = "
        terms = [f"({w:.4f} * x{i+1})" for i, w in enumerate(self.coef_)]
        equation += " + ".join(terms) + f" + {self.intercept_:.4f}"
        
        most_important_idx = np.argmax(np.abs(self.coef_))
        
        return (
            "--- GlassBox Explanation: Linear Regression ---\n"
            f"Equation: {equation}\n"
            f"Most influential feature: Feature {most_important_idx + 1} "
            f"(Weight: {self.coef_[most_important_idx]:.4f})\n"
            f"Baseline prediction (Intercept): {self.intercept_:.4f}"
        )