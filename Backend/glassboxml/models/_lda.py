import numpy as np

from glassboxml.core._base_model import GlassBoxModel

class LDA(GlassBoxModel):
    """
    Transparent Linear Discriminant Analysis (LDA).
    A supervised dimensionality reduction tool that maximizes class separability.
    """

    def __init__(self,n_components):
        super().__init__()
        self.loss_history = None
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.classes_ = None

    def check_assumptions(self, X, y):
        self.failure_modes = []
        n_features = X.shape[1]
        self.classes_ = np.unique(y)
        max_components = len(self.classes_) - 1

        # Checking for dimensionality limit
        if self.n_components > max_components:
            self.failure_modes.append(
                f"Mathematical Limit Reached: You have {len(self.classes_)} classes. "
                f"LDA can only project down to (Classes - 1) dimensions. "
                f"You requested {self.n_components}, but the maximum is {max_components}."
            )

        # Check for Perfect Multi-Collinearity
        # If features are perfectly correlated, the Within-Class scatter matrix cannot be inverted!

        correlation_matrix = np.corrcoef(X,rowvar=False)
        if n_features > 1 and np.any(np.triu(np.abs(correlation_matrix), k=1) > 0.99):
            self.failure_modes.append(
                "Severe multicollinearity detected. The Within-Class scatter matrix (S_W) "
                "might be un-invertible (singular). We will use a pseudo-inverse, but results may be unstable."
            )

        return self.failure_modes
    
    def fit(self,X,y):
        self._store_dataset_stats(X)
        self.check_assumptions(X, y)
        
        n_features = X.shape[1]
        mean_overall = np.mean(X, axis=0)
        
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        # Calculate Scatter Matrices
        for c in self.classes_:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            n_c = X_c.shape[0]
            
            # Within-Class Scatter (S_W)
            # How much does Class C deviate from its own mean?
            diff_W = X_c - mean_c
            S_W += np.dot(diff_W.T, diff_W)
            
            # Between-Class Scatter (S_B)
            # How far is Class C's mean from the global overall mean?
            diff_B = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_c * np.dot(diff_B, diff_B.T)

        # Solve the Generalized Eigenvalue Problem: (S_W^-1 * S_B) * v = lambda * v
        # We use pinv (pseudo-inverse) to protect against singular matrices
        A = np.dot(np.linalg.pinv(S_W), S_B)
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # Eigenvalues might be slightly complex due to numerical precision, make them real
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        # Sort descending
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select top n_components
        # We cap it mathematically at min(requested, classes - 1) just in case
        valid_components = min(self.n_components, len(self.classes_) - 1)
        self.components_ = eigenvectors[:, :valid_components].T


        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        if total_variance > 0:
            self.explained_variance_ratio_ = eigenvalues[:valid_components] / total_variance
        else:
            self.explained_variance_ratio_ = np.zeros(valid_components)
            
        self.is_fitted = True
        self.training_error = "N/A (Supervised Dimensionality Reduction)"


    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("GlassBox Error: Model is not fitted yet. Call .fit() first.")
        return np.dot(X, self.components_.T)
    
    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X)
    
    def predict(self, X):
        raise NotImplementedError(
            "LDA is being used here as a dimensionality reduction tool. "
            "Use .transform(X) to squash your data, then pass it to a Classifier!"
        )
    
    def explain(self):
        if not self.is_fitted:
            return "Model is not fitted."
            
        cumulative_variance = np.sum(self.explained_variance_ratio_) * 100
        n_comps = self.components_.shape[0]
        
        explanation = (
            "--- GlassBox Explanation: Linear Discriminant Analysis (LDA) ---\n"
            f"Classes Detected: {len(self.classes_)}\n"
            f"Reduced Dimensions: {n_comps}\n"
            f"Class Separability Captured: {cumulative_variance:.2f}%\n\n"
            "Interpretation: LDA found the specific axes in space that maximize the distance "
            "between different class clusters while keeping the points within each cluster tightly packed."
        )
        return explanation