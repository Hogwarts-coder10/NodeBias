import numpy as np
from glassboxml.core._base_model import GlassBoxModel

class SVM(GlassBoxModel):
    """
    Transparent Support Vector Machine (Linear).
    Uses Gradient Descent to minimize Hinge Loss and maximize the margin.
    """
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        super().__init__()
        self.loss_history = None
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.classes_ = None

    def check_assumptions(self, X, y):
        # SVMs are heavily distance-based!
        self.failure_modes = []
        variances = np.var(X, axis=0)
        if np.max(variances) / (np.min(variances) + 1e-9) > 10:
            self.failure_modes.append(
                "[WARNING] SVMs are highly sensitive to unscaled data! "
                "The margin will be completely skewed by the feature with the largest variance. "
                "Always use StandardScaler before fitting an SVM."
            )
        return self.failure_modes

    def fit(self, X, y):
        self._store_dataset_stats(X)
        self.check_assumptions(X, y)
        
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("GlassBox Error: This basic SVM only supports binary classification (2 classes).")
            
        # THE MATH TRICK: SVM math REQUIRES labels to be -1 and 1, not 0 and 1!
        # If we use 0, the multiplication in the Hinge Loss equation collapses.
        y_ = np.where(y <= 0, -1, 1)
        
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        print(f"Training SVM (Iterating {self.n_iters} times to find the widest street)...")
        
        # Gradient Descent for Hinge Loss
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Check if the point is correctly classified AND outside the margin
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    # Point is safe! Just pull the weights slightly to maximize margin (Regularization)
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    # Point is inside the margin or misclassified! (Hinge Loss active)
                    dw = 2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    db = y_[idx]
                    
                # Update weights and bias
                self.w -= self.lr * dw
                self.b -= self.lr * db
                
        self.is_fitted = True
        self.training_error = "N/A (Hinge Loss Minimized)"

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("GlassBox Error: Model is not fitted yet. Call .fit() first.")
            
        # Equation of the line: y = sign(w*x - b)
        approx = np.dot(X, self.w) - self.b
        predicted_labels = np.sign(approx)
        
        # Convert the -1/1 math outputs back to the original 0/1 labels
        return np.where(predicted_labels == -1, self.classes_[0], self.classes_[1])

    def explain(self):
        if not self.is_fitted:
            return "Model is not fitted."
            
        explanation = (
            "--- GlassBox Explanation: Linear SVM ---\n"
            f"Weights (Normal Vector to Hyperplane): {self.w}\n"
            f"Bias (Offset from Origin): {self.b:.4f}\n\n"
            "Interpretation: The SVM drew a rigid line (hyperplane) through the data space. "
            "It optimized the weights to not only separate the classes, but to make the 'street' "
            "between them as wide as mathematically possible."
        )
        return explanation
    


class SupportVectorRegressor(GlassBoxModel):
    """
    Transparent Support Vector Regressor (Linear SVR).
    Uses Gradient Descent to fit an Epsilon-Tube over continuous data.
    """
    def __init__(self, learning_rate=0.01, lambda_param=0.01, epsilon=0.1, n_iters=1000):
        super().__init__()
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.epsilon = epsilon  # The width of our "street" (the tube)
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def check_assumptions(self, X, y):
        self.failure_modes = []
        variances = np.var(X, axis=0)
        if np.max(variances) / (np.min(variances) + 1e-9) > 10:
            self.failure_modes.append(
                "[WARNING] SVR geometry is highly sensitive to unscaled data! "
                "The epsilon tube will be distorted. Always use StandardScaler first."
            )
        return self.failure_modes

    def fit(self, X, y):
        self._store_dataset_stats(X)
        self.check_assumptions(X, y)
        
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        print(f"Training SVR (Iterating {self.n_iters} times to fit the Epsilon-Tube)...")
        
        # Gradient Descent for Spsilon-Insensitive Loss
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # 1. Make a prediction
                prediction = np.dot(x_i, self.w) + self.b
                
                # 2. Calculate the exact mathematical error
                error = y[idx] - prediction
                
                # 3. Check if the point is OUTSIDE the tube
                if error >= self.epsilon:
                    # Point is above the tube (Underestimated)
                    dw = 2 * self.lambda_param * self.w - x_i
                    db = -1
                elif error <= -self.epsilon:
                    # Point is below the tube (Overestimated)
                    dw = 2 * self.lambda_param * self.w + x_i
                    db = 1
                else:
                    # Point is safely INSIDE the tube! (No penalty applied)
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                    
                # 4. Update the weights to tilt/shift the tube
                self.w -= self.lr * dw
                self.b -= self.lr * db
                
        self.is_fitted = True
        self.training_error = "N/A (Epsilon-Insensitive Loss Minimized)"

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("GlassBox Error: Model is not fitted yet. Call .fit() first.")
            
        # Equation of the line: y = w*x + b
        return np.dot(X, self.w) + self.b

    def explain(self):
        if not self.is_fitted:
            return "Model is not fitted."
            
        explanation = (
            "--- GlassBox Explanation: Linear SVR ---\n"
            f"Weights (Slope of the Tube): {self.w}\n"
            f"Bias (Y-Intercept): {self.b:.4f}\n"
            f"Epsilon (Tube Width): ±{self.epsilon}\n\n"
            "Interpretation: The SVR drew a continuous straight 'tube' through the data. "
            "Any data points that fell inside the tube were ignored. The model only updated "
            "its slope to capture the points that spilled outside the margins!"
        )
        return explanation