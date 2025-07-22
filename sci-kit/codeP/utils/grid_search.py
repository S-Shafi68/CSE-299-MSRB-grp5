"""
Grid Search and Hyperparameter Tuning
"""

import numpy as np
from itertools import product
from logger import get_logger

logger = get_logger("grid_search")

class GridSearchCV:
    """
    Grid search with cross-validation for hyperparameter tuning.
    
    Parameters:
    -----------
    estimator : object
        The estimator to tune
    param_grid : dict
        Dictionary of parameters to search
    cv : int or cross-validation generator, default=5
        Cross-validation splitting strategy
    scoring : str, default='accuracy'
        Scoring method
    refit : bool, default=True
        Whether to refit the best estimator on the whole dataset
    """
    
    def __init__(self, estimator, param_grid, cv=5, scoring='accuracy', refit=True):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        
        # Results
        self.best_estimator_ = None
        self.best_score_ = None
        self.best_params_ = None
        self.cv_results_ = None
    
    def _generate_param_combinations(self):
        """Generate all combinations of parameters."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        for combination in product(*values):
            yield dict(zip(keys, combination))
    
    def fit(self, X, y):
        """
        Fit the grid search.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        from .validation import cross_val_score, KFold
        
        # Setup cross-validation
        if isinstance(self.cv, int):
            cv = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        else:
            cv = self.cv
        
        # Results storage
        results = {
            'params': [],
            'mean_test_score': [],
            'std_test_score': [],
            'rank_test_score': []
        }
        
        param_combinations = list(self._generate_param_combinations())
        logger.info(f"Fitting {len(param_combinations)} parameter combinations...")
        
        # Try each parameter combination
        for params in param_combinations:
            # Create estimator with current parameters
            estimator = type(self.estimator)(**params)
            
            # Perform cross-validation
            scores = cross_val_score(estimator, X, y, cv=cv, scoring=self.scoring)
            
            # Store results
            results['params'].append(params)
            results['mean_test_score'].append(np.mean(scores))
            results['std_test_score'].append(np.std(scores))
        
        # Rank results
        mean_scores = np.array(results['mean_test_score'])
        ranking = np.argsort(-mean_scores)  # Descending order
        results['rank_test_score'] = [0] * len(ranking)
        for i, rank in enumerate(ranking):
            results['rank_test_score'][rank] = i + 1
        
        # Find best parameters
        best_idx = np.argmax(mean_scores)
        self.best_params_ = results['params'][best_idx]
        self.best_score_ = results['mean_test_score'][best_idx]
        self.cv_results_ = results
        
        # Refit with best parameters
        if self.refit:
            self.best_estimator_ = type(self.estimator)(**self.best_params_)
            self.best_estimator_.fit(X, y)
        
        logger.info(f"Best parameters: {self.best_params_}")
        logger.info(f"Best score: {self.best_score_:.4f}")
        
        return self
    
    def predict(self, X):
        """Make predictions using the best estimator."""
        if self.best_estimator_ is None:
            raise ValueError("GridSearchCV must be fitted before making predictions")
        
        return self.best_estimator_.predict(X)
    
    def score(self, X, y):
        """Score using the best estimator."""
        if self.best_estimator_ is None:
            raise ValueError("GridSearchCV must be fitted before scoring")
        
        return self.best_estimator_.score(X, y)

class RandomizedSearchCV:
    """
    Randomized search with cross-validation for hyperparameter tuning.
    
    Parameters:
    -----------
    estimator : object
        The estimator to tune
    param_distributions : dict
        Dictionary of parameters to search
    n_iter : int, default=10
        Number of parameter combinations to try
    cv : int or cross-validation generator, default=5
        Cross-validation splitting strategy
    scoring : str, default='accuracy'
        Scoring method
    refit : bool, default=True
        Whether to refit the best estimator on the whole dataset
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, estimator, param_distributions, n_iter=10, cv=5, 
                 scoring='accuracy', refit=True, random_state=None):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.random_state = random_state
        
        # Results
        self.best_estimator_ = None
        self.best_score_ = None
        self.best_params_ = None
        self.cv_results_ = None
    
    def _sample_parameters(self):
        """Sample parameters from distributions."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        for _ in range(self.n_iter):
            params = {}
            for param_name, param_values in self.param_distributions.items():
                if isinstance(param_values, list):
                    params[param_name] = np.random.choice(param_values)
                else:
                    # Assume it's a distribution or range
                    params[param_name] = np.random.choice(param_values)
            
            yield params
    
    def fit(self, X, y):
        """
        Fit the randomized search.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        from .validation import cross_val_score, KFold
        
        # Setup cross-validation
        if isinstance(self.cv, int):
            cv = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        else:
            cv = self.cv
        
        # Results storage
        results = {
            'params': [],
            'mean_test_score': [],
            'std_test_score': [],
            'rank_test_score': []
        }
        
        logger.info(f"Fitting {self.n_iter} random parameter combinations...")
        
        # Try each parameter combination
        for params in self._sample_parameters():
            # Create estimator with current parameters
            estimator = type(self.estimator)(**params)
            
            # Perform cross-validation
            scores = cross_val_score(estimator, X, y, cv=cv, scoring=self.scoring)
            
            # Store results
            results['params'].append(params)
            results['mean_test_score'].append(np.mean(scores))
            results['std_test_score'].append(np.std(scores))
        
        # Rank results
        mean_scores = np.array(results['mean_test_score'])
        ranking = np.argsort(-mean_scores)  # Descending order
        results['rank_test_score'] = [0] * len(ranking)
        for i, rank in enumerate(ranking):
            results['rank_test_score'][rank] = i + 1
        
        # Find best parameters
        best_idx = np.argmax(mean_scores)
        self.best_params_ = results['params'][best_idx]
        self.best_score_ = results['mean_test_score'][best_idx]
        self.cv_results_ = results
        
        # Refit with best parameters
        if self.refit:
            self.best_estimator_ = type(self.estimator)(**self.best_params_)
            self.best_estimator_.fit(X, y)
        
        logger.info(f"Best parameters: {self.best_params_}")
        logger.info(f"Best score: {self.best_score_:.4f}")
        
        return self
    
    def predict(self, X):
        """Make predictions using the best estimator."""
        if self.best_estimator_ is None:
            raise ValueError("RandomizedSearchCV must be fitted before making predictions")
        
        return self.best_estimator_.predict(X)
    
    def score(self, X, y):
        """Score using the best estimator."""
        if self.best_estimator_ is None:
            raise ValueError("RandomizedSearchCV must be fitted before scoring")
        
        return self.best_estimator_.score(X, y)

# Parameter validation utilities
def validate_parameters(estimator, param_grid):
    """
    Validate parameter grid for an estimator.
    
    Parameters:
    -----------
    estimator : object
        The estimator to validate parameters for
    param_grid : dict
        Dictionary of parameters to validate
        
    Returns:
    --------
    bool
        True if all parameters are valid
    """
    valid_params = set(estimator.get_params().keys())
    
    for param_name in param_grid.keys():
        if param_name not in valid_params:
            raise ValueError(f"Invalid parameter '{param_name}' for estimator {type(estimator).__name__}")
    
    return True
