import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Calculates the accuracy score for classification.
    Accuracy = (Correct Predictions) / (Total Predictions)
    """
    # np.mean on a boolean array automatically converts True to 1 and False to 0
    accuracy = np.mean(y_true == y_pred)
    return accuracy

def classification_report(y_true, y_pred):
    """
    Generates a quick text summary of the model's accuracy.
    """
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total
    
    report = (
        "--- GlassBox Evaluation Report ---\n"
        f"Total Test Samples : {total}\n"
        f"Correct Predictions: {correct}\n"
        f"Mistakes Made      : {total - correct}\n"
        f"Final Accuracy     : {accuracy:.2%}\n"
        "----------------------------------"
    )
    return report