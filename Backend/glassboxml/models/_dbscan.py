import numpy as np
from glassboxml.core._base_model import GlassBoxModel

class DBSCAN(GlassBoxModel):
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
    Finds core points of high density and expands clusters from them using BFS.
    """

    def __init__(self,eps = 0.5,min_samples = 5):
        super().__init__()
        self.loss_history = None
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.n_clusters = 0
        self.n_noise_ = 0

    
    def check_assumptions(self, X, y=None):
        """
        Checks if the data violates DBSCAN's core physical requirements.
        """
        self.failure_modes = []
        
        # Distance-based algorithms fail if features aren't scaled
        variances = np.var(X, axis=0)
        max_var = np.max(variances)
        min_var = np.min(variances)
        
        # If the max variance is > 10x the min variance, the Epsilon radius is highly distorted
        if max_var / (min_var + 1e-9) > 10:
            self.failure_modes.append(
                "[WARNING] High variance difference detected! DBSCAN relies on physical distance (Epsilon). "
                "If features are unscaled, the search radius becomes an oval, ruining the density check. "
                "Please run StandardScaler first."
            )
        return self.failure_modes
    
    def _get_neighbours(self,X,point_idx):
        """
        Draws the Epsilon circle and returns the indices of everyone inside.
        """
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0].tolist()
    
    def fit(self, X, y=None):
        """
        Executes the Breadth-First Search to map dense territories.
        """

        # 1. Check our framework assumptions first
        warnings = self.check_assumptions(X)
        for warning in warnings:
            print(warning)

        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1)
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        print(f"DBSCAN: Scanning {n_samples} points (Radius={self.eps}, Density Threshold={self.min_samples})...")

        for i in range(n_samples):
            if visited[i]:
                continue
            
            visited[i] = True
            neighbors = self._get_neighbours(X, i)
            
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1 # Fails the density test (Noise)
            else:
                # Passes the test! Start the BFS Chain Reaction
                cluster_id += 1
                self.labels_[i] = cluster_id
                
                j = 0
                while j < len(neighbors):
                    neighbor_idx = neighbors[j]
                    
                    if not visited[neighbor_idx]:
                        visited[neighbor_idx] = True
                        new_neighbors = self._get_neighbours(X, neighbor_idx)
                        
                        # If neighbor is also dense, enqueue their neighbors to our BFS list
                        if len(new_neighbors) >= self.min_samples:
                            neighbors.extend(new_neighbors)
                    
                    # Claim previously isolated points as border points
                    if self.labels_[neighbor_idx] == -1:
                        self.labels_[neighbor_idx] = cluster_id
                        
                    j += 1

        self.n_clusters_ = cluster_id
        self.n_noise_ = list(self.labels_).count(-1)
        self.is_fitted = True
        print(f"  -> Chain reaction complete! Found {self.n_clusters_} distinct dense clusters.")

    def predict(self, X):
        """
        Prevents standard predict() calls since DBSCAN groups existing data dynamically.
        """

        raise NotImplementedError(
            "GlassBox Error: DBSCAN is transductive. It cannot easily 'predict' new incoming data points "
            "without re-running the entire BFS chain. Read the `labels_` attribute instead!"
        )
    
    def explain(self):
        """
        Translates the BFS mathematical output into a human-readable logistics report.
        """

        if not self.is_fitted:
            return "The model has not been trained yet. Call .fit() first!"
            
        return (
            "--- GlassBox Explanation: DBSCAN ---\n"
            f"Clusters Discovered: {self.n_clusters_}\n"
            f"Outliers (Noise) Ignored: {self.n_noise_}\n\n"
            "Interpretation:\n"
            "The algorithm acted like a pathfinder looking for dense pockets of activity. "
            f"Using an Epsilon search radius of {self.eps}, it hopped from point to point, organically mapping out "
            "winding territories as long as the density stayed high. Any completely isolated coordinates "
            "were mathematically ignored and labeled as noise (-1)."
        )