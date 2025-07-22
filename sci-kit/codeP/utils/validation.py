"""
Cross-Validation and Model Validation Utilities
"""

import numpy as np
from logger import get_logger

logger = get_logger("validation")

class KFold:
    """
    K-Fold cross-validation iterator.
    
    Parameters:
    -----------
    n_splits : int, default=5
        Number of folds
    shuffle : bool, default=False
        Whether to shuffle the data before splitting
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
    
    def split(self, X, y=None):
        """
        Generate train/test splits.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,), optional
            Target values
            
        Yields:
        -------
        train_idx : array
            Training indices
        test_idx : array
            Test indices
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)
        
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size if i < self.n_splits - 1 else n_samples
            
            test_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            
            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None):
        """Get number of splits."""
        return self.n_splits

class StratifiedKFold:
    """
    Stratified K-Fold cross-validation iterator.
    
    Parameters:
    -----------
    n_splits : int, default=5
        Number of folds
    shuffle : bool, default=False
        Whether to shuffle the data before splitting
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
    
    def split(self, X, y):
        """
        Generate stratified train/test splits.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
            
        Yields:
        -------
        train_idx : array
            Training indices
        test_idx : array
            Test indices
        """
        n_samples = X.shape[0]
        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = len(classes)
        
        # Group indices by class
        class_indices = [[] for _ in range(n_classes)]
        for idx, class_idx in enumerate(y_indices):
            class_indices[class_idx].append(idx)
        
        # Shuffle indices within each class
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            for class_idx_list in class_indices:
                np.random.shuffle(class_idx_list)
        
        # Calculate fold sizes for each class
        class_fold_sizes = []
        for class_idx_list in class_indices:
            n_class_samples = len(class_idx_list)
            fold_size = n_class_samples // self.n_splits
            class_fold_sizes.append(fold_size)
        
        # Generate folds
        for fold in range(self.n_splits):
            train_idx = []
            test_idx = []
            
            for class_idx, class_idx_list in enumerate(class_indices):
                fold_size = class_fold_sizes[class_idx]
                start = fold * fold_size
                end = start + fold_size if fold < self.n_splits - 1 else len(class_idx_list)
                
                test_idx.extend(class_idx_list[start:end])
                train_idx.extend(class_idx_list[:start] + class_idx_list[end:])
            
            yield np.array(train_idx), np.array(test_idx)
    
    def get_n_splits(self, X=None, y=None):
        """Get number of splits."""
        return self.n_splits

def cross_val_score(estimator, X, y, cv=None, scoring='accuracy'):
    """
    Evaluate a score by cross-validation.
    
    Parameters:
    -----------
    estimator : object
        The estimator to evaluate
    X : array-like, shape (n_samples, n_features)
        Training data
    y : array-like, shape (n_samples,)
        Target values
    cv : int or cross-validation generator, default=None
        Cross-validation splitting strategy
    scoring : str, default='accuracy'
        Scoring method
        
    Returns:
    --------
    scores : array
        Cross-validation scores
    """
    if cv is None:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
    elif isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    scores = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Clone estimator
        estimator_clone = type(estimator)(**estimator.get_params())
        
        # Fit and predict
        estimator_clone.fit(X_train, y_train)
        
        if scoring == 'accuracy':
            score = estimator_clone.score(X_test, y_test)
        elif scoring == 'f1':
            from utils.metrics import calculate_f1
            y_pred = estimator_clone.predict(X_test)
            score = calculate_f1(y_test, y_pred, average='weighted')
        else:
            raise ValueError(f"Unknown scoring method: {scoring}")
        
        scores.append(score)
    
    return np.array(scores)

def cross_validate(estimator, X, y, cv=None, scoring='accuracy', return_train_score=False):
    """
    Evaluate estimator by cross-validation with more detailed output.
    
    Parameters:
    -----------
    estimator : object
        The estimator to evaluate
    X : array-like, shape (n_samples, n_features)
        Training data
    y : array-like, shape (n_samples,)
        Target values
    cv : int or cross-validation generator, default=None
        Cross-validation splitting strategy
    scoring : str, default='accuracy'
        Scoring method
    return_train_score : bool, default=False
        Whether to return training scores
        
    Returns:
    --------
    results : dict
        Dictionary with test scores and optionally train scores
    """
    if cv is None:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
    elif isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    test_scores = []
    train_scores = []
    fit_times = []
    score_times = []
    
    import time
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Clone estimator
        estimator_clone = type(estimator)(**estimator.get_params())
        
        # Fit and time
        start_time = time.time()
        estimator_clone.fit(X_train, y_train)
        fit_time = time.time() - start_time
        fit_times.append(fit_time)
        
        # Score and time
        start_time = time.time()
        test_score = estimator_clone.score(X_test, y_test)
        score_time = time.time() - start_time
        score_times.append(score_time)
        
        test_scores.append(test_score)
        
        if return_train_score:
            train_score = estimator_clone.score(X_train, y_train)
            train_scores.append(train_score)
    
    results = {
        'test_score': np.array(test_scores),
        'fit_time': np.array(fit_times),
        'score_time': np.array(score_times)
    }
    
    if return_train_score:
        results['train_score'] = np.array(train_scores)
    
    return results

def model_comparison(models, X, y, cv=None, scoring='accuracy'):
    """
    Compare multiple models using cross-validation.
    
    Parameters:
    -----------
    models : dict
        Dictionary of model_name: model_instance
    X : array-like, shape (n_samples, n_features)
        Training data
    y : array-like, shape (n_samples,)
        Target values
    cv : int or cross-validation generator, default=None
        Cross-validation splitting strategy
    scoring : str, default='accuracy'
        Scoring method
        
    Returns:
    --------
    results : dict
        Dictionary with results for each model
    """
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, 
                                  return_train_score=True)
        
        results[model_name] = {
            'test_scores': cv_results['test_score'],
            'train_scores': cv_results['train_score'],
            'test_mean': np.mean(cv_results['test_score']),
            'test_std': np.std(cv_results['test_score']),
            'train_mean': np.mean(cv_results['train_score']),
            'train_std': np.std(cv_results['train_score']),
            'fit_time_mean': np.mean(cv_results['fit_time']),
            'score_time_mean': np.mean(cv_results['score_time'])
        }
    
    return results
