import numpy as np
import matplotlib.pyplot as plt

def build_confusion_matrix(y_true, y_pred, labels=None):
    """
    Builds an N x N grid where rows are True Labels and columns are Predicted Labels.
    """
    if labels is None:
        # Find all unique classes in both true and predicted arrays
        labels = np.unique(np.concatenate((y_true, y_pred)))
        
    n_labels = len(labels)
    matrix = np.zeros((n_labels, n_labels), dtype=int)
    
    # Map the class names (like '0', '1', '2') to grid indices (0, 1, 2)
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    # Tally up every single prediction
    for true_val, pred_val in zip(y_true, y_pred):
        true_idx = label_to_index[true_val]
        pred_idx = label_to_index[pred_val]
        matrix[true_idx, pred_idx] += 1
        
    return matrix, labels

def plot_confusion_matrix(y_true, y_pred, title="GlassBox Confusion Matrix"):
    """
    Renders the Confusion Matrix as a beautiful heatmap.
    """
    matrix, labels = build_confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Draw the heatmap (Blues colormap makes the heavy numbers dark blue)
    cax = ax.matshow(matrix, cmap='Blues')
    fig.colorbar(cax)
    
    # Label the axes
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    
    # Move the X-axis labels to the bottom for easier reading
    ax.xaxis.set_ticks_position('bottom')
    
    plt.xlabel("Predicted Label (What the model guessed)", fontsize=11, labelpad=10)
    plt.ylabel("True Label (The actual answer)", fontsize=11, labelpad=10)
    plt.title(title, fontsize=14, pad=15)
    
    # Loop over the grid and print the actual numbers inside the boxes
    thresh = matrix.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            count = matrix[i, j]
            # If the box is dark blue, use white text. If light blue, use black text.
            color = "white" if count > thresh else "black"
            
            # Make the correct predictions (diagonal) bold!
            weight = "bold" if i == j else "normal"
            
            ax.text(j, i, str(count), va='center', ha='center', 
                    color=color, fontsize=14, weight=weight)
            
    plt.tight_layout()
    plt.show()