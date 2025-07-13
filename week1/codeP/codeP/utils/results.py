"""
Results saving functionality for the ML library recreation project.
Handles saving model outputs, visualizations, and benchmarks.
"""

import json
import pickle
import csv
from pathlib import Path
from datetime import datetime
import logging
import numpy as np


class ResultsSaver:
    """Handles saving and loading of results and models."""
    
    def __init__(self, output_dir="RESULTS"):
        """
        Initialize ResultsSaver.
        
        Args:
            output_dir (str): Base directory for saving results
        """
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary output directories."""
        directories = [
            self.output_dir,
            self.output_dir / "model_outputs",
            self.output_dir / "visualizations", 
            self.output_dir / "benchmarks",
            self.output_dir / "models"  # For saved model objects
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        self.logger.info(f"Created output directories in {self.output_dir}")
    
    def save_results(self, results_data, experiment_name):
        """
        Save experiment results to JSON file.
        
        Args:
            results_data (dict): Dictionary containing results
            experiment_name (str): Name for the experiment
        """
        # Add timestamp
        results_data['timestamp'] = datetime.now().isoformat()
        results_data['experiment_name'] = experiment_name
        
        # Save to JSON
        json_path = self.output_dir / "model_outputs" / f"{experiment_name}_{self._get_timestamp()}.json"
        
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=self._json_serializer)
            
        self.logger.info(f"Results saved to {json_path}")
        
        # Also save to CSV for easy analysis
        self._save_results_csv(results_data, experiment_name)
    
    def _save_results_csv(self, results_data, experiment_name):
        """Save results to CSV format for easy analysis."""
        csv_path = self.output_dir / "model_outputs" / "experiment_log.csv"
        
        # Flatten the results data
        flat_data = self._flatten_dict(results_data)
        
        # Check if file exists to write headers
        file_exists = csv_path.exists()
        
        with open(csv_path, 'a', newline='') as f:
            if flat_data:
                writer = csv.DictWriter(f, fieldnames=flat_data.keys())
                
                if not file_exists:
                    writer.writeheader()
                    
                writer.writerow(flat_data)
                
        if not file_exists:
            self.logger.info(f"Created experiment log: {csv_path}")
    
    def save_model(self, model, model_name, experiment_name):
        """
        Save trained model to disk.
        
        Args:
            model: Trained model object
            model_name (str): Name of the model
            experiment_name (str): Name of the experiment
        """
        model_path = self.output_dir / "models" / f"{experiment_name}_{model_name}_{self._get_timestamp()}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        self.logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, model_path):
        """
        Load trained model from disk.
        
        Args:
            model_path (str or Path): Path to the saved model
            
        Returns:
            Loaded model object
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        self.logger.info(f"Model loaded from {model_path}")
        return model
    
    def save_predictions(self, y_true, y_pred, experiment_name, set_name="test"):
        """
        Save predictions and true values for analysis.
        
        Args:
            y_true (array): True values
            y_pred (array): Predicted values
            experiment_name (str): Name of the experiment
            set_name (str): Name of the set (train/test/validation)
        """
        predictions_data = {
            'y_true': y_true.tolist() if hasattr(y_true, 'tolist') else list(y_true),
            'y_pred': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
            'set_name': set_name,
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat()
        }
        
        pred_path = self.output_dir / "model_outputs" / f"{experiment_name}_{set_name}_predictions_{self._get_timestamp()}.json"
        
        with open(pred_path, 'w') as f:
            json.dump(predictions_data, f, indent=2)
            
        self.logger.info(f"Predictions saved to {pred_path}")
        
        # Also save as CSV for easy plotting
        csv_path = self.output_dir / "model_outputs" / f"{experiment_name}_{set_name}_predictions_{self._get_timestamp()}.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['y_true', 'y_pred'])
            for true_val, pred_val in zip(y_true, y_pred):
                writer.writerow([true_val, pred_val])
    
    def save_benchmark(self, benchmark_data, benchmark_name):
        """
        Save benchmark results.
        
        Args:
            benchmark_data (dict): Benchmark results
            benchmark_name (str): Name of the benchmark
        """
        benchmark_data['timestamp'] = datetime.now().isoformat()
        benchmark_data['benchmark_name'] = benchmark_name
        
        bench_path = self.output_dir / "benchmarks" / f"{benchmark_name}_{self._get_timestamp()}.json"
        
        with open(bench_path, 'w') as f:
            json.dump(benchmark_data, f, indent=2, default=self._json_serializer)
            
        self.logger.info(f"Benchmark saved to {bench_path}")
    
    def load_results(self, experiment_name=None, latest_only=True):
        """
        Load previous results.
        
        Args:
            experiment_name (str): Specific experiment name to load
            latest_only (bool): Load only the latest results
            
        Returns:
            dict or list: Results data
        """
        results_dir = self.output_dir / "model_outputs"
        
        if experiment_name:
            # Load specific experiment
            pattern = f"{experiment_name}_*.json"
        else:
            # Load all JSON results
            pattern = "*.json"
        
        json_files = list(results_dir.glob(pattern))
        
        if not json_files:
            self.logger.warning(f"No results found matching pattern: {pattern}")
            return [] if not latest_only else None
        
        if latest_only:
            # Return the most recent file
            latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, 'r') as f:
                return json.load(f)
        else:
            # Return all matching results
            results = []
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    results.append(json.load(f))
            return results
    
    def _get_timestamp(self):
        """Get timestamp string for file naming."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return str(obj)
    
    def _flatten_dict(self, d, parent_key='', sep='_'):
        """Flatten nested dictionary for CSV export."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def get_summary_stats(self):
        """Get summary statistics of all experiments."""
        csv_path = self.output_dir / "model_outputs" / "experiment_log.csv"
        
        if not csv_path.exists():
            return "No experiments found."
        
        # Read CSV and provide basic stats
        experiments = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            experiments = list(reader)
        
        if not experiments:
            return "No experiments found."
        
        summary = {
            'total_experiments': len(experiments),
            'models_used': list(set(exp.get('model', 'unknown') for exp in experiments)),
            'datasets_used': list(set(exp.get('dataset', 'unknown') for exp in experiments)),
            'latest_experiment': max(experiments, key=lambda x: x.get('timestamp', ''))
        }
        
        return summary