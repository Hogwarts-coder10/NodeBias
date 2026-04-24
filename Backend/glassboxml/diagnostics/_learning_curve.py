import numpy as np
import matplotlib.pyplot as plt

class LearningCurveAnalyzer:
    """
    Diagnoses Bias (Underfitting) and Variance (Overfitting) by plotting 
    model performance across increasing amounts of training data.
    """
    
    @staticmethod
    def generate_curve(model, X, y, is_classifier=False, n_splits=5, test_size=0.2):
        """
        Trains the model on increasingly larger subsets of the data and 
        records the error on both the training set and a held-out validation set.
        """
        # Shuffle the data
        indices = np.random.permutation(len(X))
        X_shuffled, y_shuffled = X[indices], y[indices]
        
        # Create a simple validation split
        split_idx = int(len(X) * (1 - test_size))
        X_train_full, X_val = X_shuffled[:split_idx], X_shuffled[split_idx:]
        y_train_full, y_val = y_shuffled[:split_idx], y_shuffled[split_idx:]
        
        train_sizes = np.linspace(0.1, 1.0, n_splits)
        train_errors = []
        val_errors = []
        
        print("Generating Learning Curve (this might take a moment)...")
        
        for size in train_sizes:
            # Take a subset of the training data
            subset_idx = int(len(X_train_full) * size)
            X_subset = X_train_full[:subset_idx]
            y_subset = y_train_full[:subset_idx]
            
            # Train the model from scratch on this subset
            model.fit(X_subset, y_subset)
            
            # Predict on the training subset and the full validation set
            y_train_pred = model.predict(X_subset)
            y_val_pred = model.predict(X_val)
            
            # Calculate Error (Misclassification rate for trees, MSE for regression)
            if is_classifier:
                train_err = np.mean(y_train_pred != y_subset)
                val_err = np.mean(y_val_pred != y_val)
            else:
                train_err = np.mean((y_train_pred - y_subset)**2)
                val_err = np.mean((y_val_pred - y_val)**2)
                
            train_errors.append(train_err)
            val_errors.append(val_err)
            
        actual_sizes = [int(len(X_train_full) * size) for size in train_sizes]
        return actual_sizes, train_errors, val_errors

    @staticmethod
    def plot_curve(train_sizes, train_errors, val_errors, title="Learning Curve"):
        """
        Visualizes the bias-variance tradeoff.
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_sizes, train_errors, 'o-', color="blue", label="Training Error", linewidth=2)
        plt.plot(train_sizes, val_errors, 'o-', color="red", label="Validation Error", linewidth=2)
        
        # Fill the gap between lines to emphasize variance
        plt.fill_between(train_sizes, train_errors, val_errors, color="red", alpha=0.1)
        
        plt.title(f"Bias-Variance Diagnostic: {title}", fontsize=14)
        plt.xlabel("Number of Training Samples", fontsize=12)
        plt.ylabel("Error Rate", fontsize=12)
        plt.legend(loc="best", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Add educational annotations
        final_gap = val_errors[-1] - train_errors[-1]
        if final_gap > np.mean(val_errors) * 0.5:
            diag_text = "[WARNING] Diagnosis: HIGH VARIANCE (Overfitting)\nThe model memorized the training data\nbut fails on new data. The gap is too wide."
        elif val_errors[-1] > 0.3: # Arbitrary high error threshold
            diag_text = "[WARNING] Diagnosis: HIGH BIAS (Underfitting)\nThe model is too simple to capture the pattern.\nBoth errors are high."
        else:
            diag_text = "[OK] Diagnosis: GOOD FIT\nTraining and validation errors converge to a low value."
            
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        plt.gca().text(0.5, 0.5, diag_text, transform=plt.gca().transAxes, fontsize=11,
                verticalalignment='center', horizontalalignment='center', bbox=props)
        
        plt.tight_layout()
        plt.show()