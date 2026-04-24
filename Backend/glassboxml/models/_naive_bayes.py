import numpy as np
from glassboxml.core._base_model import GlassBoxModel

class GaussianNaiveBayes(GlassBoxModel):
    def __init__(self):
        super().__init__()
        self.loss_history = None
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialize arrays to store our mean, variance, and prior probabilities
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        # Calculate the statistics for each class
        for idx, c in enumerate(self.classes):
            # Isolate the data points that belong to class 'c'
            X_c = X[y == c]
            
            # The Likelihood math: Mean and Variance
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            
            # The Prior math: Frequency of this class
            self.priors[idx] = X_c.shape[0] / float(n_samples)
            
        self.is_fitted = True

    def predict(self, X):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])

            mean = self.mean[idx]
            var = self.var[idx] + 1e-9

            numerator = -((X - mean) ** 2) / (2 * var)
            log_likelihood = np.sum(numerator - np.log(np.sqrt(2 * np.pi * var)), axis=1)

            posterior = prior + log_likelihood
            posteriors.append(posterior)

        posteriors = np.array(posteriors).T
        return self.classes[np.argmax(posteriors, axis=1)]
        
    def _pdf(self, class_idx, x):
        """Gaussian Probability Density Function"""
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        
        # The raw math for a normal distribution curve
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def explain(self):
        if not self.is_fitted:
            return "Model is not fitted yet."
        
        explanation = f"--- GlassBox Explanation: Gaussian Naive Bayes ---\n"
        explanation += f"Classes Detected: {self.classes}\n"
        explanation += f"Prior Probabilities: {dict(zip(self.classes, np.round(self.priors, 3)))}\n"
        explanation += "Interpretation: The model calculates the probability of a data point belonging to each class using Gaussian (bell curve) distributions, assuming all features are independent."
        return explanation