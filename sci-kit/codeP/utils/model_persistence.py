"""
Model Persistence and Serialization Utilities
"""

import pickle
import json
import os
from pathlib import Path
import numpy as np
from logger import get_logger

logger = get_logger("model_persistence")

class ModelPersistence:
    """
    Utility class for saving and loading ML models.
    """
    
    @staticmethod
    def save_model(model, filepath, metadata=None):
        """
        Save a trained model to disk.
        
        Parameters:
        -----------
        model : object
            Trained model to save
        filepath : str
            Path where to save the model
        metadata : dict, optional
            Additional metadata to save with the model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare model data
        model_data = {
            'model': model,
            'model_type': type(model).__name__,
            'metadata': metadata or {},
            'version': '1.0'
        }
        
        # Save model
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            
            # Save metadata as JSON for easy inspection
            metadata_path = filepath.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump({
                    'model_type': model_data['model_type'],
                    'metadata': model_data['metadata'],
                    'version': model_data['version'],
                    'file_size': os.path.getsize(filepath)
                }, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    @staticmethod
    def load_model(filepath):
        """
        Load a model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        model : object
            Loaded model
        metadata : dict
            Model metadata
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            logger.info(f"Model loaded from {filepath}")
            
            return model_data['model'], model_data.get('metadata', {})
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    @staticmethod
    def list_saved_models(directory="saved_models"):
        """
        List all saved models in a directory.
        
        Parameters:
        -----------
        directory : str
            Directory to search for models
            
        Returns:
        --------
        models : list
            List of model information dictionaries
        """
        directory = Path(directory)
        
        if not directory.exists():
            return []
        
        models = []
        for model_file in directory.glob("*.pkl"):
            metadata_file = model_file.with_suffix('.json')
            
            model_info = {
                'filename': model_file.name,
                'path': str(model_file),
                'size': os.path.getsize(model_file),
                'modified': os.path.getmtime(model_file)
            }
            
            # Load metadata if available
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    model_info.update(metadata)
                except:
                    pass
            
            models.append(model_info)
        
        return sorted(models, key=lambda x: x['modified'], reverse=True)
