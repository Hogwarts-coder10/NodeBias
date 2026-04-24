import numpy as np
from typing import Optional

# Adjust path if your base class is elsewhere
from glassboxml.core._base_model import GlassBoxModel 

class Perceptron(GlassBoxModel):
    """
    The classic Rosenblatt Perceptron (1957).
    
    A single-layer linear binary classifier that updates its weights 
    based on a step-function error gradient. The fundamental building 
    block of all modern Neural Networks.
    """
    def __init__(
        self, 
        learning_rate: float = 0.01, 
        n_iters: int = 1000, 
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.random_state = random_state
        
        # Level 2: Clean initialization of state attributes
        self.weights_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None
        self.is_fitted: bool = False

    def _step_function(self, x: np.ndarray) -> np.ndarray:
        """The Activation Function: Returns 1 if x >= 0, else 0."""
        return np.where(x >= 0, 1, 0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        # Level 1: Strict Input Validation
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays")
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")

        n_samples, n_features = X.shape
        
        # Initialize weights and bias using our thread-safe RNG
        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.normal(0, 0.01, n_features) # Start with tiny random weights
        self.bias_ = 0.0
        
        # The true Perceptron only handles binary classification {0, 1}
        # We ensure y is strictly binary for the math to hold up
        y_binary = np.where(y > 0, 1, 0)

        # -------------------------------------------------------------
        # The Core Algorithm: Train the Neuron
        # -------------------------------------------------------------
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # 1. Forward Pass: Dot product of inputs & weights, plus bias
                linear_output = np.dot(x_i, self.weights_) + self.bias_
                
                # 2. Activation: Run it through the step function
                y_predicted = self._step_function(linear_output)
                
                # 3. The Perceptron Update Rule
                # If prediction is correct (y == y_pred), update is 0.
                # If prediction is wrong, it pushes weights toward the correct class.
                update = self.learning_rate * (y_binary[idx] - y_predicted)
                
                self.weights_ += update * x_i
                self.bias_ += update
                
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Call fit() before predict().")
            
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise TypeError("X must be a 2D numpy array")
        
        # Forward pass on new data
        linear_output = np.dot(X, self.weights_) + self.bias_
        return self._step_function(linear_output)

    def explain(self) -> str:
        if not self.is_fitted:
            return "Model is not fitted yet."
        
        explanation = "--- GlassBox Explanation: Perceptron ---\n"
        explanation += f"Learned Weights: {np.round(self.weights_, 4)}\n"
        explanation += f"Learned Bias: {self.bias_:.4f}\n"
        explanation += "Interpretation: The grandfather of neural networks. It draws a single straight line (hyperplane) to separate two classes, updating its weights mathematically every time it makes a mistake during training."
        return explanation