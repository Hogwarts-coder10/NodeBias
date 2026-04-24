import numpy as np
from glassboxml.core._base_model import GlassBoxModel

class KNNClassifier(GlassBoxModel):
    """
    Transparent K-Nearest Neighbors Classifier.
    A non-parametric, lazy learning algorithm based purely on geometric distance.
    """
    def __init__(self, k=3, p=2):
        super().__init__()
        self.loss_history = None
        self.k = k
        self.p = p # 1: Manhattan, 2: Euclidean

    def check_assumptions(self, X, y):
        self.failure_modes = []
        n_features = X.shape[1]

        if self.k % 2 == 0:
            self.failure_modes.append(
                f"Parity Warning: k is set to an even number ({self.k}). In binary classification, "
                "this can lead to voting ties. Consider using an odd k."
            )

        if n_features > 15:
            self.failure_modes.append(
                f"Curse of Dimensionality: High dimensionality detected ({n_features} features). "
                "In high-dimensional space, all points become nearly equidistant, rendering "
                "nearest neighbors meaningless. Consider PCA first."
            )

        variance = np.var(X, axis=0)
        max_var, min_var = np.max(variance), np.min(variance)

        if min_var > 0 and (max_var / min_var) > 10:
            self.failure_modes.append(
                "Unscaled features detected. The feature with the massive variance "
                "will completely dominate the distance calculation. Please use StandardScaler."
            )

        return self.failure_modes

    def fit(self, X, y):
        self._store_dataset_stats(X)
        self.check_assumptions(X, y)

        # Lazy Learning: Just store the data into the "memory bank"
        self.X_train = X
        self.y_train = y
        self.is_fitted = True

        self.training_error = "N/A (Lazy Learner)"
        self.loss_history = ["N/A"]

    def _compute_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("GlassBox Error: Model is not fitted yet. Call .fit() first.")

        y_pred = []
        for x_new in X:
            distances = [self._compute_distance(x_new, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]

            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            y_pred.append(unique_labels[np.argmax(counts)])

        return np.array(y_pred)

    def explain(self):
        """Translates the lazy-learning configuration into human readable text."""
        if not self.is_fitted:
            return "Model is not fitted."

        metric = "Manhattan (Grid-like)" if self.p == 1 else "Euclidean (Straight-line)" if self.p == 2 else f"Minkowski (p={self.p})"
        samples = self.dataset_stats_.get('n_samples', 0)

        return (
            "--- GlassBox Explanation: K-Nearest Neighbors ---\n"
            f"Neighbors Polled (k): {self.k}\n"
            f"Distance Metric: {metric}\n"
            f"Memory Bank: {samples} samples actively stored.\n"
            "Interpretation: KNN has no equation. It predicts by measuring the distance between "
            "the new point and every point in its memory bank, taking a majority vote among the closest neighbors."
        )
