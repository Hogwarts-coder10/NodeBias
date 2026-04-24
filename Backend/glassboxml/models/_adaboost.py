import numpy as np
from glassboxml.core._base_model import GlassBoxModel

class DecisionStump:
    """
    A 'Weak Learner' for AdaBoost. It is a tree with a max depth of exactly 1.
    It draws a single straight horizontal or vertical line.
    """
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column < self.threshold] = -1
            predictions = -predictions
            
        return predictions

class AdaBoostClassifier(GlassBoxModel):
    """
    Transparent AdaBoost (Adaptive Boosting) Classifier.
    Chains together weak Decision Stumps, mathematically forcing each new stump
    to focus on the data points that the previous stumps got wrong.
    """
    
    def __init__(self, n_clf=50):
        super().__init__()
        self.n_clf = n_clf
        self.clfs = []
        self.classes_ = None

    def check_assumptions(self, X, y):
        self.failure_modes = []
        # AdaBoost is actually highly susceptible to massive outliers
        # because it will give them infinite weight trying to fix them!
        return self.failure_modes

    def fit(self, X, y):
        self._store_dataset_stats(X)
        self.check_assumptions(X, y)
        
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("GlassBox Error: This basic AdaBoost only supports binary classification (2 classes).")
            
        # Like SVMs, AdaBoost math requires labels to be -1 and 1
        y_ = np.where(y <= 0, -1, 1)
        
        n_samples, n_features = X.shape
        
        # Step 1: Initialize all data points with equal weights
        w = np.full(n_samples, (1 / n_samples))
        
        self.clfs = []
        print(f"Boosting: Training {self.n_clf} sequential Decision Stumps...")
        
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')
            
            # Find the best feature and threshold that minimizes the WEIGHTED error
            for feat_i in range(n_features):
                X_column = X[:, feat_i]
                thresholds = np.unique(X_column)
                
                for threshold in thresholds:
                    # Test flipping the greater-than / less-than signs (polarity)
                    for polarity in [1, -1]:
                        predictions = np.ones(n_samples)
                        if polarity == 1:
                            predictions[X_column < threshold] = -1
                        else:
                            predictions[X_column < threshold] = -1
                            predictions = -predictions
                            
                        # Calculate weighted error (sum of weights for wrong predictions)
                        misclassified_weights = w[y_ != predictions]
                        error = sum(misclassified_weights)
                        
                        if error < min_error:
                            min_error = error
                            clf.polarity = polarity
                            clf.threshold = threshold
                            clf.feature_idx = feat_i
            
            # Step 2: Calculate Alpha (how much voting power this stump gets)
            EPS = 1e-10 # Prevent division by zero
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))
            
            # Step 3: Update and normalize the weights!
            # If y_ and predictions match, they multiply to 1 (weight drops).
            # If they don't match, they multiply to -1 (weight spikes!).
            predictions = clf.predict(X)
            w *= np.exp(-clf.alpha * y_ * predictions)
            w /= np.sum(w)
            
            self.clfs.append(clf)
            
        self.is_fitted = True
        self.training_error = "N/A (Adaptive Weighting)"

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("GlassBox Error: Model is not fitted yet.")
            
        # Every stump makes a prediction (-1 or 1), multiplied by its Alpha (voting power)
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        
        # Sum them all up!
        y_pred = np.sum(clf_preds, axis=0)
        
        # Convert signs back to original classes
        y_pred = np.sign(y_pred)
        return np.where(y_pred == -1, self.classes_[0], self.classes_[1])

    def explain(self):
        if not self.is_fitted:
            return "Model is not fitted."
            
        return (
            "--- GlassBox Explanation: AdaBoost ---\n"
            f"Ensemble Size: {self.n_clf} Sequential Decision Stumps\n\n"
            "Interpretation: The model chained together incredibly weak, 1-depth trees. "
            "After every stump drew its single line, the algorithm multiplied the 'weight' "
            "of the misclassified data points, forcing the next stump to focus on the hardest "
            "parts of the dataset. The final prediction is a weighted vote based on the accuracy "
            "of each stump."
        )