import numpy as np
from abc import ABC, abstractmethod

class GlassBoxModel(ABC):
    """
    The base class for all GlassBox-ML models.
    Enforces a Scikit-Learn style API with educational diagnostics.
    """
    def __init__(self):
        # Standard library naming conventions
        self.coef_ = None
        self.intercept_ = None
        self.is_fitted = False
        
        # Educational & Diagnostic trackers
        self.failure_modes = []
        self.loss_history = []
        self.gradient_history = []
        self.training_error = None
        self.dataset_stats_ = {}

    def _store_dataset_stats(self, X):
        """Records the statistical footprint of the training data."""
        self.dataset_stats_ = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'mean': np.mean(X, axis=0),
            'variance': np.var(X, axis=0),
            'min': np.min(X, axis=0),
            'max': np.max(X, axis=0)
        }

    @abstractmethod
    def fit(self, X, y):
        """Trains the model and updates self.coef_ and self.intercept_."""
        pass

    @abstractmethod
    def predict(self, X):
        """Returns the final prediction."""
        pass

    def decision_function(self, X):
        """
        Returns the raw mathematical output before any thresholding or activation.
        For linear models, this is: X * coef_ + intercept_
        """
        if not self.is_fitted:
            raise ValueError("GlassBox Error: Model is not fitted yet. Call .fit() first.")
        return np.dot(X, self.coef_) + self.intercept_

    @abstractmethod
    def explain(self):
        """Returns a human-readable interpretation of the learned weights."""
        pass
        
    def check_assumptions(self, X, y):
        """Analyzes data for potential mathematical failures."""
        return []

    def _record_step(self, epoch_loss, epoch_gradients):
        """Logs the learning journey for visualization."""
        self.loss_history.append(epoch_loss)
        self.gradient_history.append(epoch_gradients)

    def diagnose(self):
        """Returns a comprehensive diagnostic report of the model's health."""
        features = self.dataset_stats_.get('n_features', 0)
        samples = self.dataset_stats_.get('n_samples', 0)
        
        return {
            "Dataset Profile": f"{samples} samples, {features} features",
            "Final Training Error": self.training_error,
            "Optimization Steps": len(self.loss_history),
            "Logged Failure Modes": self.failure_modes if self.failure_modes else "None detected"
        }