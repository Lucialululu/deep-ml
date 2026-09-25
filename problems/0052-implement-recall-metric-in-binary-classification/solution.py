import numpy as np
from sklearn.metrics import confusion_matrix


def recall(y_true, y_pred):
    """
    Calculate the recall metric for binary classification.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)
    
    Returns:
        Recall value as a float
    """
    # Your code here
    conf_matrix = confusion_matrix(y_true, y_pred)
    TN = conf_matrix[0][0]
    TP = conf_matrix[1][1]
    FN = conf_matrix[1][0]
    
    recall = TP / (TP + FN)

    return recall
