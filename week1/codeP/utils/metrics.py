"""
Basic evaluation metrics for the ML library recreation project.
Implements common regression and classification metrics from scratch.
"""

import numpy as np
import logging


def mean_squared_error(y_true, y_pred):
    """
    Calculate Mean Squared Error.
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        
    Returns:
        float: Mean squared error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        
    Returns:
        float: Mean absolute error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    Calculate R² (coefficient of determination).
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        
    Returns:
        float: R² score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    # Sum of squares of residuals
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Total sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Handle edge case where all y_true values are the same
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)


def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        
    Returns:
        float: Root mean squared error
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_accuracy(y_true, y_pred):
    """
    Calculate classification accuracy.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        
    Returns:
        float: Accuracy score (between 0 and 1)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    return np.mean(y_true == y_pred)


def calculate_precision(y_true, y_pred, average='binary', labels=None):
    """
    Calculate precision score.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        average (str): Averaging strategy ('binary', 'macro', 'micro', 'weighted')
        labels (array-like): Labels to include in calculation
        
    Returns:
        float or array: Precision score(s)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    precisions = []
    
    for label in labels:
        # True positives
        tp = np.sum((y_true == label) & (y_pred == label))
        # False positives
        fp = np.sum((y_true != label) & (y_pred == label))
        
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        
        precisions.append(precision)
    
    precisions = np.array(precisions)
    
    if average == 'binary':
        if len(labels) != 2:
            raise ValueError("Binary average requires exactly 2 classes")
        return precisions[1]  # Return precision for positive class
    elif average == 'macro':
        return np.mean(precisions)
    elif average == 'micro':
        # Calculate global TP and FP
        tp_total = np.sum([np.sum((y_true == label) & (y_pred == label)) for label in labels])
        fp_total = np.sum([np.sum((y_true != label) & (y_pred == label)) for label in labels])
        
        if tp_total + fp_total == 0:
            return 0.0
        return tp_total / (tp_total + fp_total)
    elif average == 'weighted':
        # Weight by support (number of true instances for each label)
        supports = [np.sum(y_true == label) for label in labels]
        return np.average(precisions, weights=supports)
    else:
        return precisions


def calculate_recall(y_true, y_pred, average='binary', labels=None):
    """
    Calculate recall score.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        average (str): Averaging strategy ('binary', 'macro', 'micro', 'weighted')
        labels (array-like): Labels to include in calculation
        
    Returns:
        float or array: Recall score(s)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    recalls = []
    
    for label in labels:
        # True positives
        tp = np.sum((y_true == label) & (y_pred == label))
        # False negatives
        fn = np.sum((y_true == label) & (y_pred != label))
        
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        
        recalls.append(recall)
    
    recalls = np.array(recalls)
    
    if average == 'binary':
        if len(labels) != 2:
            raise ValueError("Binary average requires exactly 2 classes")
        return recalls[1]  # Return recall for positive class
    elif average == 'macro':
        return np.mean(recalls)
    elif average == 'micro':
        # Calculate global TP and FN
        tp_total = np.sum([np.sum((y_true == label) & (y_pred == label)) for label in labels])
        fn_total = np.sum([np.sum((y_true == label) & (y_pred != label)) for label in labels])
        
        if tp_total + fn_total == 0:
            return 0.0
        return tp_total / (tp_total + fn_total)
    elif average == 'weighted':
        # Weight by support (number of true instances for each label)
        supports = [np.sum(y_true == label) for label in labels]
        return np.average(recalls, weights=supports)
    else:
        return recalls


def calculate_f1(y_true, y_pred, average='binary', labels=None):
    """
    Calculate F1 score.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        average (str): Averaging strategy ('binary', 'macro', 'micro', 'weighted')
        labels (array-like): Labels to include in calculation
        
    Returns:
        float or array: F1 score(s)
    """
    precision = calculate_precision(y_true, y_pred, average=average, labels=labels)
    recall = calculate_recall(y_true, y_pred, average=average, labels=labels)
    
    if isinstance(precision, np.ndarray):
        # Handle array case
        f1_scores = np.zeros_like(precision)
        mask = (precision + recall) > 0
        f1_scores[mask] = 2 * (precision[mask] * recall[mask]) / (precision[mask] + recall[mask])
        return f1_scores
    else:
        # Handle scalar case
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute confusion matrix.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        labels (array-like): Labels to include in matrix
        
    Returns:
        array: Confusion matrix of shape (n_classes, n_classes)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    n_labels = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    # Initialize confusion matrix
    cm = np.zeros((n_labels, n_labels), dtype=int)
    
    # Fill confusion matrix
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in label_to_idx and pred_label in label_to_idx:
            true_idx = label_to_idx[true_label]
            pred_idx = label_to_idx[pred_label]
            cm[true_idx, pred_idx] += 1
    
    return cm


def classification_report(y_true, y_pred, labels=None, target_names=None):
    """
    Build a text report showing main classification metrics.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        labels (array-like): Labels to include in report
        target_names (array-like): Display names for labels
        
    Returns:
        str: Text summary of precision, recall, F1 score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    if target_names is None:
        target_names = [str(label) for label in labels]
    
    # Calculate metrics for each class
    precisions = calculate_precision(y_true, y_pred, average=None, labels=labels)
    recalls = calculate_recall(y_true, y_pred, average=None, labels=labels)
    f1_scores = calculate_f1(y_true, y_pred, average=None, labels=labels)
    
    # Calculate support (number of occurrences of each class)
    supports = [np.sum(y_true == label) for label in labels]
    
    # Build report string
    report = f"{'':>12} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}\n\n"
    
    for i, (name, precision, recall, f1, support) in enumerate(
        zip(target_names, precisions, recalls, f1_scores, supports)
    ):
        report += f"{name:>12} {precision:>9.2f} {recall:>9.2f} {f1:>9.2f} {support:>9}\n"
    
    # Add macro and weighted averages
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)
    
    weighted_precision = np.average(precisions, weights=supports)
    weighted_recall = np.average(recalls, weights=supports)
    weighted_f1 = np.average(f1_scores, weights=supports)
    
    total_support = sum(supports)
    
    report += f"\n{'macro avg':>12} {macro_precision:>9.2f} {macro_recall:>9.2f} {macro_f1:>9.2f} {total_support:>9}\n"
    report += f"{'weighted avg':>12} {weighted_precision:>9.2f} {weighted_recall:>9.2f} {weighted_f1:>9.2f} {total_support:>9}\n"
    
    return report


# Convenience function to calculate all regression metrics at once
def regression_metrics(y_true, y_pred):
    """
    Calculate all common regression metrics.
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        
    Returns:
        dict: Dictionary containing MSE, MAE, RMSE, and R² scores
    """
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': calculate_rmse(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


# Convenience function to calculate all classification metrics at once
def classification_metrics(y_true, y_pred, average='binary'):
    """
    Calculate all common classification metrics.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        average (str): Averaging strategy
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1 scores
    """
    return {
        'accuracy': calculate_accuracy(y_true, y_pred),
        'precision': calculate_precision(y_true, y_pred, average=average),
        'recall': calculate_recall(y_true, y_pred, average=average),
        'f1': calculate_f1(y_true, y_pred, average=average)
    }