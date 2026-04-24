import numpy as np
from glassboxml.core._base_model import GlassBoxModel
from glassboxml.models._decision_tree import DecisionTreeRegressor

class GradientBoostingRegressor(GlassBoxModel):
    """
    Transparent Gradient Boosting Regressor.
    Chains together Decision Trees where each new tree is trained to predict 
    and correct the residual errors of the combined ensemble before it.
    """
    def __init__(self, n_trees=100, learning_rate=0.1, max_depth=3):
        super().__init__()
        self.n_trees = n_trees
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.base_prediction = None

    def check_assumptions(self, X, y):
        self.failure_modes = []
        if self.max_depth > 5:
            self.failure_modes.append(
                "[WARNING] High max_depth detected. Gradient Boosting uses shallow trees "
                "(often called 'weak learners'). Deep trees will cause the model to over-correct "
                "and instantly memorize the noise (Overfitting)."
            )
        return self.failure_modes

    def fit(self, X, y):
        self._store_dataset_stats(X)
        self.check_assumptions(X, y)
        
        self.trees = []
        
        # Step 1: The Base Model (just predict the mean of y)
        self.base_prediction = np.mean(y)
        
        # The running prediction of our ensemble starts as just the mean
        current_predictions = np.full(len(y), self.base_prediction)
        
        print(f"Gradient Boosting: Training {self.n_trees} sequential error-correcting trees...")
        
        for i in range(self.n_trees):
            # Step 2: Calculate the mathematical mistakes (Residuals)
            # For Mean Squared Error, the negative gradient is literally just (Actual - Predicted)
            residuals = y - current_predictions
            
            # Step 3: Train a tree to predict the MISTAKES, not the actual target!
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # Step 4: Add the new tree's predictions to our running total, scaled by learning rate
            update_step = tree.predict(X)
            current_predictions += self.lr * update_step
            
            self.trees.append(tree)
            
        self.is_fitted = True
        self.training_error = "N/A (Residual Error Minimized)"

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("GlassBox Error: Model is not fitted yet.")
            
        # Start with the base mean prediction
        y_pred = np.full(X.shape[0], self.base_prediction)
        
        # Add the scaled predictions from every single error-correcting tree
        for tree in self.trees:
            y_pred += self.lr * tree.predict(X)
            
        return y_pred

    def explain(self):
        if not self.is_fitted:
            return "Model is not fitted."
            
        return (
            "--- GlassBox Explanation: Gradient Boosting Regressor ---\n"
            f"Ensemble Size: {self.n_trees} Trees\n"
            f"Learning Rate: {self.lr}\n\n"
            "Interpretation: The model started by guessing the exact same average value for every point. "
            "Then, it calculated how wrong that guess was. It trained a tree to predict that exact error, "
            "and added a fraction of that tree's prediction to the total. It repeated this process, with "
            "every new tree fixing the tiny mistakes left behind by the trees before it."
        )
    


class GradientBoostingClassifier(GlassBoxModel):
    """
    Transparent Gradient Boosting Classifier.
    Trains Regression Trees to predict the probability residuals (Log-Loss gradients),
    gradually pushing the log-odds boundary to perfectly separate the classes.
    """
    def __init__(self, n_trees=100, learning_rate=0.1, max_depth=3):
        super().__init__()
        self.n_trees = n_trees
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.base_log_odds = None
        self.classes_ = None

    def _sigmoid(self, x):
        # Clip x to prevent overflow in exp
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def check_assumptions(self, X, y):
        self.failure_modes = []
        if self.max_depth > 5:
            self.failure_modes.append(
                "[WARNING] High max_depth detected. Gradient Boosting uses shallow trees. "
                "Deep trees will overfit to the residuals and memorize the noise."
            )
        return self.failure_modes

    def fit(self, X, y):
        self._store_dataset_stats(X)
        self.check_assumptions(X, y)
        
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("GlassBox Error: This basic GBM only supports binary classification (2 classes).")
            
        # Ensure labels are exactly 0 and 1 for log-loss math
        y_ = np.where(y == self.classes_[1], 1, 0)
        
        self.trees = []
        
        # Step 1: Base Prediction (Log-Odds of the positive class)
        p_pos = np.mean(y_)
        p_pos = np.clip(p_pos, 1e-10, 1 - 1e-10) # Prevent divide-by-zero
        self.base_log_odds = np.log(p_pos / (1.0 - p_pos))
        
        # The running raw log-odds prediction starts as the base_log_odds
        F_x = np.full(len(y_), self.base_log_odds)
        
        print(f"GBM Classifier: Training {self.n_trees} regression trees on probability residuals...")
        
        for i in range(self.n_trees):
            # Step 2: Convert log-odds to probabilities
            probabilities = self._sigmoid(F_x)
            
            # Step 3: Calculate the pseudo-residuals (Negative Gradient of Log-Loss)
            # This is elegantly simple: Actual (0 or 1) - Predicted Probability (0.0 to 1.0)
            residuals = y_ - probabilities
            
            # Step 4: Train a REGRESSION tree to predict these continuous errors
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # Step 5: Update the running log-odds
            update_step = tree.predict(X)
            F_x += self.lr * update_step
            
            self.trees.append(tree)
            
        self.is_fitted = True
        self.training_error = "N/A (Log-Loss Minimized)"

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("GlassBox Error: Model is not fitted yet.")
            
        # Start with the base log-odds
        F_x = np.full(X.shape[0], self.base_log_odds)
        
        # Add the scaled predictions from every tree
        for tree in self.trees:
            F_x += self.lr * tree.predict(X)
            
        # Convert final log-odds back to probabilities
        probabilities = self._sigmoid(F_x)
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        # Threshold at 50%
        y_pred = np.where(probabilities >= 0.5, self.classes_[1], self.classes_[0])
        return y_pred

    def explain(self):
        if not self.is_fitted:
            return "Model is not fitted."
            
        return (
            "--- GlassBox Explanation: Gradient Boosting Classifier ---\n"
            f"Ensemble Size: {self.n_trees} Regression Trees\n"
            f"Learning Rate: {self.lr}\n\n"
            "Interpretation: Even though this is a Classifier, it uses Regression Trees! "
            "It started by predicting the log-odds of the majority class. Then, it calculated "
            "the probability of each point and found the 'residual' (Actual Label 0/1 - Predicted Probability). "
            "It trained a Regression Tree to predict that continuous error, pushing the probability "
            "closer to 1.0 for the positive class and 0.0 for the negative class."
        )