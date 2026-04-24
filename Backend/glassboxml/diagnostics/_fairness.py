import numpy as np

class FairnessAnalyzer:
    """
    Diagnoses whether a model performs equally well across different subgroups
    of the data, preventing biased predictions and disparate impact.
    """
    
    @staticmethod
    def check_disparate_impact(y_true, y_pred, sensitive_attribute, group_names=None):
        """
        Compares the model's accuracy between different groups defined by a 
        sensitive attribute.
        """
        groups = np.unique(sensitive_attribute)
        report = ["--- GlassBox Fairness & Disparate Impact Report ---"]
        
        overall_acc = np.mean(y_true == y_pred)
        report.append(f"Overall Model Accuracy: {overall_acc:.2%}\n")
        
        group_metrics = {}
        for i, g in enumerate(groups):
            mask = (sensitive_attribute == g)
            group_acc = np.mean(y_true[mask] == y_pred[mask])
            group_metrics[g] = group_acc
            
            name = group_names[i] if group_names else f"Group {g}"
            report.append(f"{name} Accuracy: {group_acc:.2%} (N={np.sum(mask)})")
            
        # Find the maximum gap in accuracy
        accuracies = list(group_metrics.values())
        max_gap = max(accuracies) - min(accuracies)
        
        report.append(f"\nMaximum Accuracy Gap Between Groups: {max_gap:.2%}")
        
        if max_gap > 0.15: # 15% threshold for warning
            report.append("[WARNING] High disparate impact detected. The model is significantly less accurate for certain subgroups.")
        else:
            report.append("[OK] Model performs relatively consistently across subgroups.")
            
        return "\n".join(report)