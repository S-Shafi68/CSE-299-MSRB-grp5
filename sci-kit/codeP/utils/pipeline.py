"""
Machine Learning Pipeline Implementation
"""

import numpy as np
from logger import get_logger

logger = get_logger("pipeline")

class Pipeline:
    """
    Pipeline of transforms with a final estimator.
    
    Parameters:
    -----------
    steps : list of tuples
        List of (name, transformer/estimator) tuples
    memory : object, default=None
        Used to cache fitted transformers
    verbose : bool, default=False
        Whether to print progress
    """
    
    def __init__(self, steps, memory=None, verbose=False):
        self.steps = steps
        self.memory = memory
        self.verbose = verbose
        
        # Validate steps
        self._validate_steps()
        
        # Fitted attributes
        self.named_steps = dict(steps)
        
        logger.debug(f"Initialized Pipeline with {len(steps)} steps")
    
    def _validate_steps(self):
        """Validate pipeline steps."""
        if len(self.steps) == 0:
            raise ValueError("Pipeline cannot be empty")
        
        for name, estimator in self.steps:
            if not isinstance(name, str):
                raise ValueError("Step names must be strings")
            if not hasattr(estimator, 'fit'):
                raise ValueError("All steps must have a 'fit' method")
        
        # Check that all but last step have transform method
        for name, estimator in self.steps[:-1]:
            if not hasattr(estimator, 'transform'):
                raise ValueError(f"Step '{name}' must have a 'transform' method")
    
    def _iter_steps(self):
        """Iterate over (name, estimator) tuples."""
        return iter(self.steps)
    
    def fit(self, X, y=None):
        """
        Fit the pipeline.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,), optional
            Target values
            
        Returns:
        --------
        self : Pipeline
            Returns self for method chaining
        """
        X_transformed = X
        
        for step_idx, (name, estimator) in enumerate(self._iter_steps()):
            if self.verbose:
                logger.info(f"Fitting step {step_idx + 1}/{len(self.steps)}: {name}")
            
            if step_idx == len(self.steps) - 1:
                # Last step - fit the final estimator
                estimator.fit(X_transformed, y)
            else:
                # Intermediate step - fit and transform
                estimator.fit(X_transformed, y)
                X_transformed = estimator.transform(X_transformed)
        
        return self
    
    def predict(self, X):
        """
        Apply transforms and predict with final estimator.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predictions
        """
        X_transformed = X
        
        # Apply all transformations
        for name, estimator in self.steps[:-1]:
            if self.verbose:
                logger.debug(f"Applying transformation: {name}")
            X_transformed = estimator.transform(X_transformed)
        
        # Make prediction with final estimator
        final_estimator = self.steps[-1][1]
        return final_estimator.predict(X_transformed)
    
    def predict_proba(self, X):
        """Apply transforms and predict probabilities with final estimator."""
        X_transformed = X
        
        # Apply all transformations
        for name, estimator in self.steps[:-1]:
            X_transformed = estimator.transform(X_transformed)
        
        # Get probabilities from final estimator
        final_estimator = self.steps[-1][1]
        if hasattr(final_estimator, 'predict_proba'):
            return final_estimator.predict_proba(X_transformed)
        else:
            raise AttributeError("Final estimator doesn't support predict_proba")
    
    def score(self, X, y):
        """Score the pipeline."""
        X_transformed = X
        
        # Apply all transformations
        for name, estimator in self.steps[:-1]:
            X_transformed = estimator.transform(X_transformed)
        
        # Score with final estimator
        final_estimator = self.steps[-1][1]
        return final_estimator.score(X_transformed, y)
    
    def transform(self, X):
        """Apply all transformations (excluding final estimator)."""
        X_transformed = X
        
        for name, estimator in self.steps[:-1]:
            X_transformed = estimator.transform(X_transformed)
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """Fit pipeline and transform the data."""
        return self.fit(X, y).transform(X)
    
    def get_params(self, deep=True):
        """Get parameters for pipeline."""
        params = {}
        
        if deep:
            for name, estimator in self.steps:
                for param_name, param_value in estimator.get_params().items():
                    params[f"{name}__{param_name}"] = param_value
        
        params.update({
            'steps': self.steps,
            'memory': self.memory,
            'verbose': self.verbose
        })
        
        return params
    
    def set_params(self, **params):
        """Set parameters for pipeline."""
        # Handle step-specific parameters
        step_params = {}
        for param_name, param_value in params.items():
            if '__' in param_name:
                step_name, param_key = param_name.split('__', 1)
                if step_name not in step_params:
                    step_params[step_name] = {}
                step_params[step_name][param_key] = param_value
            else:
                setattr(self, param_name, param_value)
        
        # Set parameters for individual steps
        for step_name, step_param_dict in step_params.items():
            if step_name in self.named_steps:
                self.named_steps[step_name].set_params(**step_param_dict)
        
        return self
    
    def __getitem__(self, ind):
        """Get a step by name or index."""
        if isinstance(ind, str):
            return self.named_steps[ind]
        return self.steps[ind][1]
    
    def __len__(self):
        """Return number of steps in pipeline."""
        return len(self.steps)
    
    def __repr__(self):
        class_name = self.__class__.__name__
        step_strings = [f"('{name}', {estimator})" for name, estimator in self.steps]
        return f"{class_name}(steps=[{', '.join(step_strings)}])"

def make_pipeline(*steps, memory=None, verbose=False):
    """
    Construct a Pipeline from the given estimators.
    
    Parameters:
    -----------
    *steps : list of estimators
        List of estimators
    memory : object, default=None
        Used to cache fitted transformers
    verbose : bool, default=False
        Whether to print progress
        
    Returns:
    --------
    pipeline : Pipeline
        Pipeline object
    """
    names = [type(estimator).__name__.lower() for estimator in steps]
    
    # Make names unique
    name_counts = {}
    unique_names = []
    for name in names:
        if name in name_counts:
            name_counts[name] += 1
            unique_names.append(f"{name}_{name_counts[name]}")
        else:
            name_counts[name] = 0
            unique_names.append(name)
    
    return Pipeline(list(zip(unique_names, steps)), memory=memory, verbose=verbose)
