import numpy as np

def confusion_matrix(y_true,y_pred):
    """
    Computes the confusion matrix to evaluate classification strategy.
    Returns: True Positives, False Positives, True Negatives, False Negatives
    """

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tp,fp,tn,fn


def accuracy(y_true,y_pred):
    """The percentage of correct predictions out of all predictions."""
    tp,fp,tn,fn = confusion_matrix(y_true,y_pred)
    total = tp+fp+tn+fn
    return (tp + tn) / total if total > 0 else 0.0

def precision(y_true,y_pred):
    """
    Out of all the times the model predicted Class 1, how often was it right?
    Crucial when False Positives are highly costly (e.g., spam filters).
    """
    tp, fp, tn, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(y_true,y_pred):
    """
    Out of all the actual Class 1 instances, how many did the model find?
    Crucial when False Negatives are highly costly (e.g., medical diagnoses).
    """
    tp, fp, tn, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score(y_true,y_pred):
    """
    The harmonic mean of Precision and Recall. 
    Best for severely imbalanced datasets.
    """

    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

def classification_report(y_true,y_pred):
    """
    The GlassBox metric evaluator. 
    Calculates metrics and explicitly warns about the Accuracy Paradox.
    """

    acc = accuracy(y_true, y_pred)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    report = (
        f"--- GlassBox Classification Report ---\n"
        f"Accuracy:  {acc:.4f}\n"
        f"Precision: {prec:.4f}\n"
        f"Recall:    {rec:.4f}\n"
        f"F1-Score:  {f1:.4f}\n"
        f"--------------------------------------\n"
    )
    
    # GlassBox Educational Warnings
    warnings = []
    
    if acc > 0.90 and (prec < 0.10 or rec < 0.10):
        warnings.append(
            "⚠️ THE ACCURACY PARADOX: Your accuracy is high, but Precision/Recall are near zero. "
            "Your model is likely just blindly guessing the majority class in a highly imbalanced dataset. "
            "Ignore Accuracy and focus on optimizing the F1-Score."
        )
        
    if prec == 0.0 and rec == 0.0:
        warnings.append(
            "⚠️ MODEL FAILURE: The model failed to correctly identify a single instance of Class 1. "
            "Check your learning rate, or verify that your dataset isn't entirely Class 0."
        )
        
    if warnings:
        report += "🔍 DIAGNOSTICS:\n" + "\n".join(warnings) + "\n"
        
    return report

def confusion_matrix(y_true, y_pred):
    """
    Builds a basic 2x2 Confusion Matrix for binary classification.
    Returns: [[True Negatives, False Positives],
              [False Negatives, True Positives]]
    """
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    return np.array([[tn, fp], [fn, tp]])