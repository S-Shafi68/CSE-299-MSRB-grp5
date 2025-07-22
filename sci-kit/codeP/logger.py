"""
Logging System for ML Library
Provides structured logging with different levels and outputs
"""
import logging
import os
import sys
from datetime import datetime
from typing import Optional

class MLLogger:
    """Custom logger for the ML library"""
    
    def __init__(self, name: str = "ml_library", log_file: Optional[str] = None, 
                 level: str = "INFO", verbose: bool = True):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler (if verbose)
        if verbose:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler (if log_file specified)
        if log_file:
            # Ensure directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def log_model_training(self, model_name: str, dataset_name: str, params: dict):
        """Log model training start"""
        self.info(f"Starting training: {model_name} on {dataset_name}")
        self.debug(f"Parameters: {params}")
    
    def log_model_results(self, model_name: str, metrics: dict, training_time: float):
        """Log model training results"""
        self.info(f"Training completed: {model_name}")
        self.info(f"Training time: {training_time:.3f} seconds")
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.info(f"{metric}: {value:.4f}")
            else:
                self.info(f"{metric}: {value}")
    
    def log_data_info(self, dataset_name: str, shape: tuple, features: list = None):
        """Log dataset information"""
        self.info(f"Loaded dataset: {dataset_name}")
        self.info(f"Shape: {shape[0]} samples, {shape[1]} features")
        if features:
            self.debug(f"Features: {features}")
    
    def log_preprocessing(self, operation: str, details: str = ""):
        """Log preprocessing operations"""
        self.info(f"Preprocessing: {operation} {details}".strip())
    
    def log_error_with_context(self, error: Exception, context: str):
        """Log error with additional context"""
        self.error(f"Error in {context}: {str(error)}")
        self.debug(f"Error type: {type(error).__name__}")

class LoggerFactory:
    """Factory for creating loggers with consistent configuration"""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str = "ml_library", config=None) -> MLLogger:
        """Get or create logger with given name"""
        if name not in cls._loggers:
            if config:
                cls._loggers[name] = MLLogger(
                    name=name,
                    log_file=config.log_file,
                    level=config.log_level,
                    verbose=config.verbose
                )
            else:
                cls._loggers[name] = MLLogger(name=name)
        
        return cls._loggers[name]
    
    @classmethod 
    def configure_logging(cls, config):
        """Configure logging system with config"""
        # Create main logger
        logger = cls.get_logger("ml_library", config)
        
        # Log startup information
        logger.info("="*50)
        logger.info("ML Library Started")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Log Level: {config.log_level}")
        logger.info(f"Results Directory: {config.results_dir}")
        logger.info("="*50)
        
        return logger

# Convenience functions
def get_logger(name: str = "ml_library") -> MLLogger:
    """Get logger instance"""
    return LoggerFactory.get_logger(name)

def setup_logging(config):
    """Setup logging system with configuration"""
    return LoggerFactory.configure_logging(config)

# Context manager for timing operations
class TimeLogger:
    """Context manager for logging execution time"""
    
    def __init__(self, logger: MLLogger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            if exc_type is None:
                self.logger.debug(f"Completed: {self.operation} ({duration:.3f}s)")
            else:
                self.logger.error(f"Failed: {self.operation} ({duration:.3f}s) - {exc_val}")