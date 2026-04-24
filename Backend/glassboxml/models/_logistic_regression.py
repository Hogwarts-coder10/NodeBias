import numpy as np
from glassboxml.core._base_model import GlassBoxModel

class LogisticRegression(GlassBoxModel):
    """Transparent Logistic Regression for binary classification."""
    
    def __init__(self, optimizer, epochs=1000, threshold=0.5, loss_function='bce'):
        super().__init__()
        self.optimizer = optimizer
        self.epochs = epochs
        self.threshold = threshold
        self.loss_function = loss_function.lower()

    def _sigmoid(self, z):
        z = np.clip(z, -250, 250) 
        return 1 / (1 + np.exp(-z))

    def check_assumptions(self, X, y):
        self.failure_modes = []
        if self.loss_function == 'mse':
            self.failure_modes.append(
                "EDUCATIONAL WARNING: You are using MSE for classification. "
                "Expect vanishing gradients and poor convergence."
            )
        class_1_ratio = np.mean(y)
        if class_1_ratio < 0.1 or class_1_ratio > 0.9:
            self.failure_modes.append(
                f"Severe Class Imbalance detected (Class 1 ratio: {class_1_ratio:.2f})."
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
            linear_model = np.dot(X, self.coef_) + self.intercept_
            y_pred = self._sigmoid(linear_model)
            
            # 2. Compute Loss & Gradients
            if self.loss_function == 'bce':
                epsilon = 1e-9
                loss = -(1 / n_samples) * np.sum(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
                grads = {
                    'dw': (1 / n_samples) * np.dot(X.T, (y_pred - y)),
                    'db': (1 / n_samples) * np.sum(y_pred - y)
                }
            elif self.loss_function == 'mse':
                loss = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
                sigmoid_derivative = y_pred * (1 - y_pred)
                error_term = (y_pred - y) * sigmoid_derivative
                grads = {
                    'dw': (1 / n_samples) * np.dot(X.T, error_term),
                    'db': (1 / n_samples) * np.sum(error_term)
                }
            
            # 3. Clean One-Liner Optimizer Update
            self.coef_, self.intercept_ = self.optimizer.update(
                self.coef_, self.intercept_, grads['dw'], grads['db']
            )
            
            # 4. Record Step
            self._record_step(epoch_loss=loss, epoch_gradients={'dw': grads['dw'].copy(), 'db': grads['db']})
            
        self.is_fitted = True
        self.training_error = self.loss_history[-1]

    def predict_proba(self, X):
        """Returns probabilities using the Sigmoid function."""
        return self._sigmoid(self.decision_function(X))

    def predict(self, X):
        """Returns discrete classes based on the threshold."""
        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(int)

    def explain(self):
        if not self.is_fitted:
            return "Model is not fitted."
            
        return (
            "--- GlassBox Explanation: Logistic Regression ---\n"
            f"Decision Boundary (Log-Odds): X * {np.round(self.coef_, 4)} + {self.intercept_:.4f} = 0\n"
            f"Decision Threshold: {self.threshold}\n"
            "Interpretation: Features with positive coefficients push the probability toward Class 1. "
            "Features with negative coefficients push the probability toward Class 0."
        )