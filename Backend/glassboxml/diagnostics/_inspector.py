import numpy as np

class DataInspector:
    """
    Transparent diagnostic tools to evaluate datasets before they touch a model.
    Helps prevent the 'Garbage In, Garbage Out' trap.
    """
    
    @staticmethod
    def detect_data_leakage(X, y, feature_names=None, threshold=0.95):
        """
        Scans for features that are suspiciously highly correlated with the target variable.
        A correlation near 1.0 often means the feature IS the target in disguise.
        """
        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
            
        leakage_warnings = []
        
        # Calculate Pearson correlation coefficient for each feature against y
        for i in range(n_features):
            feature_col = X[:, i]
            
            # Avoid divide-by-zero if a feature is perfectly constant
            if np.std(feature_col) == 0:
                continue
                
            correlation = np.abs(np.corrcoef(feature_col, y)[0, 1])
            
            if correlation >= threshold:
                leakage_warnings.append(
                    f"🚨 LEAKAGE ALERT: '{feature_names[i]}' has a {correlation:.2%} "
                    f"correlation with the target! The model will just copy this feature "
                    f"and learn nothing else. Remove it before training."
                )
                
        if not leakage_warnings:
            return "✅ GlassBox Check: No obvious data leakage detected. Safe to proceed."
            
        return "\n".join(leakage_warnings)

    @staticmethod
    def detect_imbalance(y, threshold=0.20):
        """
        Checks if the classes are so unbalanced that the model might just 
        guess the majority class every single time.
        """
        classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        report = ["--- Class Balance Report ---"]
        for cls, count in zip(classes, counts):
            percentage = count / total_samples
            report.append(f"Class {cls}: {percentage:.1%} ({count} samples)")
            
            if percentage < threshold:
                report.append(
                    f"  ⚠️ WARNING: Class {cls} is severely underrepresented. "
                    f"The model may struggle to learn how to predict it."
                )
                
        return "\n".join(report)