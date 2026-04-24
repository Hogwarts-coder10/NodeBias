import numpy as np
from glassboxml.core._base_model import GlassBoxModel

class PCA(GlassBoxModel):
    """
    Transparent Principal Component Analysis
    An unsupervised linear algebra algorithm for dimensionality reduction.
    """

    def __init__(self,n_components):
        super().__init__()
        self.loss_history = None
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def check_assumptions(self, X, y=None):
        self.failure_modes = []
        n_samples,n_features = X.shape

        # Checking Component limit
        if self.n_components > n_features:
            self.failure_modes.append(
                f"Cannot request {self.n_components} components from data with only {n_features} features. "
                f"n_components must be <= n_features."
            )

        # Checking scale which is crucial for PCA
        variances = np.var(X, axis=0)
        max_var, min_var = np.max(variances), np.min(variances)

        if min_var > 0 and (max_var / min_var) > 5:
            self.failure_modes.append(
                "Unscaled features detected! PCA searches for the axis of maximum variance. "
                "If one feature is measured in thousands and another in decimals, PCA will completely "
                "ignore the decimal feature. Please use StandardScaler first."
            )

        return self.failure_modes
    
    def fit(self,X,y = None):
        """
        In Unsupervised learning y is ignored.
        """

        self._store_dataset_stats(X)
        self.check_assumptions(X)

        n_samples = X.shape[0]

        # Center the mean (Mean Centering)
        self.mean_ = np.mean(X,axis=0)
        X_centered = X - self.mean_

        # Find Covariance matrix
        # rowvar = False,  means columns are variables, rows are observations
        cov_matrix = np.cov(X_centered,rowvar=False)
        
        # Eigen Decomposition
        eigenvalues,eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Sort in descending order (highest variance first)
        # np.argsort returns ascending, so we reverse it with [::-1]
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # 5. Store the top 'n_components'
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]

        # Calculate the ratio (how much of the total pie does each component hold?)
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        self.is_fitted = True
        self.training_error = "N/A (Unsupervised)"

    def transform(self, X):
        """
        Projects the data onto the newly discovered Principal Components.
        """

        if not self.is_fitted:
            raise ValueError("GlassBox Error: Model is not fitted yet. Call .fit() first.")
            
        # Always center the new data using the TRAINING mean
        X_centered = X - self.mean_
        
        # Project using matrix multiplication
        return np.dot(X_centered, self.components_.T)
    
    def fit_transform(self, X, y=None):
        """
        Fits the model and immediately transforms the data.
        """

        self.fit(X, y)
        return self.transform(X)
    
    def predict(self, X):
        """
        Override to provide an educational error.
        """
        
        raise NotImplementedError(
            "PCA is an unsupervised dimensionality reduction tool, not a predictor. "
            "Use .transform(X) to squash your data, then pass that into a Classifier!"
        )
    
    def explain(self):
        if not self.is_fitted:
            return "Model is not fitted."
            
        cumulative_variance = np.sum(self.explained_variance_ratio_) * 100
        
        explanation = (
            "--- GlassBox Explanation: Principal Component Analysis (PCA) ---\n"
            f"Original Dimensions: {self.dataset_stats_['n_features']}\n"
            f"Reduced Dimensions: {self.n_components}\n"
            f"Cumulative Information Kept: {cumulative_variance:.2f}%\n\n"
            "Component Breakdown:\n"
        )
        
        for i, ratio in enumerate(self.explained_variance_ratio_):
            explanation += f"  PC{i+1}: {ratio * 100:.2f}% of total variance\n"
            
        explanation += (
            "\nInterpretation: By applying a linear transformation, we have squashed the data "
            f"down to {self.n_components} dimensions while retaining {cumulative_variance:.2f}% "
            "of the original mathematical footprint."
        )
        return explanation
    