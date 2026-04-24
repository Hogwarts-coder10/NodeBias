import numpy as np
from typing import Optional, Any

# Adjust path if your base class is elsewhere
from glassboxml.core._base_model import GlassBoxModel 

class SparseRandomProjection(GlassBoxModel):
    """
    Sparse Random Projection using the Achlioptas distribution.
    
    Reduces dimensionality while approximately preserving pairwise distances 
    between data points (Johnson-Lindenstrauss lemma). Uses a highly sparse 
    matrix for blazing-fast matrix multiplication.
    """
    def __init__(
        self, 
        n_components: int = 100, 
        density: Any = 'auto', 
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.loss_history = None
        self.n_components = n_components
        self.density = density
        self.random_state = random_state
        
        # Level 2: Clean initialization of state attributes
        self.components_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.density_: Optional[float] = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SparseRandomProjection":
        # Level 1: Strict Input Validation
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")

        n_samples, n_features = X.shape
        self.n_features_ = n_features  # Level 2: Store learned attributes

        # 1. Calculate the sparsity (density) of the matrix
        if self.density == 'auto':
            calculated_density = 1.0 / np.sqrt(n_features)
        else:
            calculated_density = float(self.density)

        # Level 1: Density edge case validation
        if calculated_density <= 0 or calculated_density > 1:
            raise ValueError("density must be in (0, 1]")
            
        self.density_ = calculated_density # Level 3: Naming consistency

        # 2. Calculate the scaling constant 'c'
        c = np.sqrt(1.0 / (self.density_ * self.n_components))

        # 3. Define the Achlioptas probability distribution
        p_pos = self.density_ / 2.0
        p_neg = self.density_ / 2.0
        p_zero = 1.0 - self.density_

        # 4. Generate the Sparse Random Matrix
        elements = [c, 0.0, -c]
        probabilities = [p_pos, p_zero, p_neg]

        # Level 1: Thread-safe, isolated Random Number Generator
        rng = np.random.default_rng(self.random_state)
        
        self.components_ = rng.choice(
            elements,
            size=(self.n_features_, self.n_components),
            p=probabilities
        )

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Level 2 & 3: Better error messaging
        if not self.is_fitted:
            raise ValueError("Call fit() before transform().")
            
        # Validate incoming transform data
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
            
        # Level 2: Prevent silent dimension mismatch bugs
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Input feature dimension mismatch. Expected {self.n_features_}, got {X.shape[1]}")

        # Level 3: Optional float32 cast for faster large-scale computation
        X = X.astype(np.float32)

        # 5. The Projection!
        return X @ self.components_

    def predict(self, X):
        raise NotImplementedError(
            "SparseRandomProjection does not support predict(). Use transform() instead."
        )

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def explain(self) -> str:
        if not self.is_fitted:
            return "Model is not fitted yet."
        
        sparsity = (self.components_ == 0).mean() * 100
        
        explanation = "--- GlassBox Explanation: Sparse Random Projection ---\n"
        explanation += f"Original Features ({self.n_features_}) reduced to: {self.n_components} dimensions.\n"
        explanation += f"Matrix Sparsity (Zeros): {sparsity:.2f}%\n"
        explanation += "Interpretation: Uses the Johnson-Lindenstrauss lemma to project high-dimensional data down into a lower space. By using a sparse random matrix, it preserves pairwise distances between data points while being computationally cheaper than PCA."
        return explanation
    
    def diagnose(self):
        return {
            "n_components": self.n_components,
            "density": self.density,
            "fitted": self.is_fitted
        }