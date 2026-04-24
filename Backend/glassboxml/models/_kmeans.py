import numpy as np
from glassboxml.core._base_model import GlassBoxModel

class KMeansClustering(GlassBoxModel):
    """
    Transparent K-Means Clustering.
    An unsupervised learning algorithm that discovers hidden groupings in unlabeled data
    by iteratively moving 'Centroids' to the mathematical center of local point clusters.
    """
    def __init__(self, k=3, max_iters=100, tol=1e-4, n_init=10):
        super().__init__()
        self.k = k
        self.max_iters = max_iters
        self.tol = tol  # Tolerance for stopping (if centroids barely move, we stop)
        self.n_init = n_init
        self.centroids = None
        self.clusters = None

    def check_assumptions(self, X, y=None):
        self.failure_modes = []
        variances = np.var(X, axis=0)
        if np.max(variances) / (np.min(variances) + 1e-9) > 10:
            self.failure_modes.append(
                "[WARNING] K-Means measures physical Euclidean distance! If your features "
                "are on different scales (e.g., GPS coordinates vs. inventory counts), the distance "
                "math will be completely warped. Always use StandardScaler first."
            )
        return self.failure_modes

    def _euclidean_distance(self, point, data):
        """Calculates the straight-line distance between a point and an array of points."""
        return np.sqrt(np.sum((data - point) ** 2, axis=1))

    def fit(self, X, y=None):
        n_samples, n_features = X.shape
        
        # Trackers for the ultimate winner across all runs
        best_inertia = float('inf')
        best_centroids = None
        best_clusters = None
        best_labels = None
        
        for init_run in range(self.n_init):
            # --- 1. Random Initial Centroids ---
            random_indices = np.random.choice(n_samples, self.k, replace=False)
            current_centroids = X[random_indices]
            
            # --- 2. The Core K-Means Loop ---
            for i in range(self.max_iters):
                current_clusters = [[] for _ in range(self.k)]
                current_labels = np.zeros(n_samples)
                
                for idx, point in enumerate(X):
                    distances = self._euclidean_distance(point, current_centroids)
                    closest_centroid_idx = np.argmin(distances)
                    current_clusters[closest_centroid_idx].append(point)
                    current_labels[idx] = closest_centroid_idx
                    
                old_centroids = np.copy(current_centroids)
                
                for cluster_idx, cluster_points in enumerate(current_clusters):
                    if len(cluster_points) == 0:
                        continue
                    current_centroids[cluster_idx] = np.mean(cluster_points, axis=0)
                    
                # Convergence Check
                if np.all(old_centroids == current_centroids):
                    break
                    
            # --- 3. Calculate Inertia (WCSS) for THIS run ---
            current_inertia = 0.0
            for cluster_idx, cluster_points in enumerate(current_clusters):
                if len(cluster_points) > 0:
                    points_array = np.array(cluster_points)
                    distances = np.linalg.norm(points_array - current_centroids[cluster_idx], axis=1)
                    current_inertia += np.sum(distances ** 2)
                    
            # --- 4. Keep the best run yet ---
            if current_inertia < best_inertia:
                best_inertia = current_inertia
                best_centroids = np.copy(current_centroids)
                best_clusters = current_clusters
                best_labels = current_labels

        # --- 5. Lock in results ---
        self.inertia_ = best_inertia
        self.centroids = best_centroids
        self.clusters = best_clusters
        self.labels_ = best_labels
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("GlassBox Error: Model is not fitted yet.")
            
        predictions = []
        for point in X:
            distances = self._euclidean_distance(point, self.centroids)
            predictions.append(np.argmin(distances))
            
        return np.array(predictions)

    def explain(self):
        if not self.is_fitted:
            return "Model is not fitted."
            
        return (
            "--- GlassBox Explanation: K-Means Clustering ---\n"
            f"Clusters (K): {self.k}\n\n"
            "Interpretation: The model grouped the completely unlabeled data into "
            f"{self.k} distinct territories. It did this by dropping anchors, drawing borders "
            "based on pure distance, and shifting the anchors until they rested in the perfect "
            "mathematical center of their respective communities."
        )