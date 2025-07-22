"""
Configuration Management System
Handles default settings and configuration for the ML library
"""
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class Config:
    """Configuration class for the ML library"""
    
    # Default paths
    data_dir: str = "DATA"
    results_dir: str = "RESULTS"
    models_dir: str = "RESULTS/models"
    plots_dir: str = "RESULTS/visualizations"
    
    # Default model parameters
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: str = "RESULTS/ml_library.log"
    
    # Output settings
    verbose: bool = True
    save_results: bool = True
    save_plots: bool = True
    
    # Performance settings
    n_jobs: int = 1
    max_iter: int = 1000
    tolerance: float = 1e-6

class ConfigManager:
    """Manages configuration loading and saving"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = Config()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file if it exists"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                    
                # Update config with loaded values
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                        
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
                print("Using default configuration")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            config_dict = {
                key: value for key, value in self.config.__dict__.items()
                if not key.startswith('_')
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=4)
                
        except Exception as e:
            print(f"Warning: Could not save config file: {e}")
    
    def get_config(self) -> Config:
        """Get current configuration"""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"Warning: Unknown configuration parameter: {key}")
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.config.data_dir,
            self.config.results_dir,
            self.config.models_dir,
            self.config.plots_dir,
            os.path.dirname(self.config.log_file)
        ]
        
        for directory in directories:
            if directory and not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                except Exception as e:
                    print(f"Warning: Could not create directory {directory}: {e}")

# Global config manager instance
config_manager = ConfigManager()

def get_config() -> Config:
    """Get the global configuration"""
    return config_manager.get_config()

def update_config(**kwargs):
    """Update the global configuration"""
    config_manager.update_config(**kwargs)

def ensure_directories():
    """Ensure all necessary directories exist"""
    config_manager.ensure_directories()