"""
app/utils/hazard_config.py
Module for loading and accessing hazard configuration data with scoring thresholds.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple


class ScoringConfig:
    """
    Main class for loading and accessing hazard configuration data.
    
    Example usage:
        config = ScoringConfig()  # Automatically finds config file
        # or
        config = ScoringConfig('path/to/config.yaml')
        
        # Get thresholds for ER hazard with 5-point scoring
        thresholds, scores = config.get_thresholds('ER', '5')
        
        # Get thresholds for LS hazard ARI with 10-point scoring
        thresholds, scores = config.get_thresholds('LS', '10', threshold_type='thresholds_ari')
        
        # Score a value
        score = config.score_value('ER', 15, '5')
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the ScoringConfig by loading the YAML configuration.
        
        Args:
            config_path: Path to the YAML configuration file. 
                        If None, automatically finds it in standard locations.
        """
        if config_path is None:
            # Automatically find config file
            self.config_path = self._find_config_path()
        else:
            self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {self.config_path}")
        
        self.config = self._load_config_with_numpy_inf(self.config_path)
        self._scoring_systems = ['3', '5', '10', '100']  # Supported scoring systems
        
    def _find_config_path(self) -> Path:
        """Find the config file in standard locations."""
        # Get the directory where this file is located
        current_file = Path(__file__).absolute()
        
        # Try different relative paths from the utils directory
        possible_paths = [
            # Relative to utils directory
            current_file.parent.parent / 'config' / 'scoring.yaml',           # app/utils/../../config/config.yaml
            current_file.parent.parent / 'scoring.yaml',                      # app/utils/../../config.yaml
            current_file.parent.parent.parent / 'config' / 'scoring.yaml',    # app/utils/../../../config/config.yaml
            
            # Relative to current working directory
            Path.cwd() / 'config' / 'scoring.yaml',
            Path.cwd().parent / 'config' / 'scoring.yaml',
            Path.cwd().parent.parent / 'config' / 'scoring.yaml',
        ]
        
        for path in possible_paths:
            if path.exists():
                print(f"Found config at: {path}")
                return path
        
        raise FileNotFoundError(
            "Could not find config.yaml in standard locations. "
            "Please provide config_path explicitly."
        )
    
    def _load_config_with_numpy_inf(self, filepath: Path) -> Dict:
        """Load YAML config and convert 'Infinity' strings to np.inf."""
        with open(filepath, 'r') as file:
            config = yaml.safe_load(file)
        
        def convert_inf(obj):
            if isinstance(obj, dict):
                return {k: convert_inf(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_inf(v) for v in obj]
            elif isinstance(obj, str) and obj.lower() in ['infinity', '.inf', 'inf']:
                return np.inf
            elif isinstance(obj, str) and obj.lower() in ['-infinity', '-.inf', '-inf']:
                return -np.inf
            else:
                return obj
        
        return convert_inf(config)
    
    def _normalize_scoring(self, scoring: Union[str, int]) -> str:
        """
        Normalize scoring input to the format used in config.
        
        Args:
            scoring: '5', '10', '3', '100' or '5_point', '10_point', etc.
        
        Returns:
            Normalized scoring string (e.g., '5_point')
        """
        scoring = str(scoring)
        if scoring.isdigit():
            return f"{scoring}_point"
        elif scoring.endswith('_point'):
            return scoring
        else:
            return f"{scoring}_point"
    
    def list_hazards(self) -> List[str]:
        """Return a list of all available hazard codes."""
        return list(self.config['hazards'].keys())
    
    def get_hazard_info(self, hazard_code: str) -> Dict:
        """
        Get basic information about a hazard.
        
        Args:
            hazard_code: e.g., 'ER', 'LS', 'CF'
        
        Returns:
            Dictionary with hazard information
        """
        if hazard_code not in self.config['hazards']:
            raise ValueError(f"Hazard '{hazard_code}' not found. Available: {self.list_hazards()}")
        
        hazard = self.config['hazards'][hazard_code]
        return {
            'code': hazard_code,
            'name': hazard.get('name', ''),
            'full_name': hazard.get('full_name', ''),
            'description': hazard.get('description', ''),
            'type': hazard.get('type', ''),
            'metrics': list(hazard.get('metrics', {}).keys())
        }
    
    def get_thresholds(self, 
                      hazard_code: str, 
                      scoring: Union[str, int] = '5',
                      metric: Optional[str] = None,
                      threshold_type: Optional[str] = None) -> Tuple[List[float], List[int]]:
        """
        Get thresholds and scores for a specific hazard and scoring system.
        
        Args:
            hazard_code: e.g., 'ER', 'LS', 'CF'
            scoring: '5', '10', '3', '100' or '5_point', '10_point', etc.
            metric: Metric name (if None, uses the first available metric)
            threshold_type: For hazards with multiple thresholds (e.g., 'thresholds_ari' for LS)
        
        Returns:
            Tuple of (thresholds list, scores list)
        
        Example:
            thresholds, scores = config.get_thresholds('ER', '5')
            thresholds, scores = config.get_thresholds('LS', '10', threshold_type='thresholds_ari')
        """
        scoring_key = self._normalize_scoring(scoring)
        
        if hazard_code not in self.config['hazards']:
            raise ValueError(f"Hazard '{hazard_code}' not found")
        
        hazard = self.config['hazards'][hazard_code]
        
        # If metric not specified, use first available
        if metric is None:
            metric = list(hazard['metrics'].keys())[0]
        
        if metric not in hazard['metrics']:
            raise ValueError(f"Metric '{metric}' not found for hazard '{hazard_code}'. Available: {list(hazard['metrics'].keys())}")
        
        metric_data = hazard['metrics'][metric]
        
        # Try different paths to find the scoring data
        scoring_data = None
        
        # Path 1: Direct 'scoring' key
        if 'scoring' in metric_data and scoring_key in metric_data['scoring']:
            scoring_data = metric_data['scoring'][scoring_key]
        
        # Path 2: 'thresholds' key
        elif 'thresholds' in metric_data and scoring_key in metric_data['thresholds']:
            scoring_data = metric_data['thresholds'][scoring_key]
        
        # Path 3: Special threshold types (for LS, etc.)
        elif threshold_type and threshold_type in metric_data:
            if scoring_key in metric_data[threshold_type]:
                scoring_data = metric_data[threshold_type][scoring_key]
        
        # Path 4: Search for any matching scoring key in metric_data
        else:
            for key in metric_data:
                if isinstance(metric_data[key], dict) and scoring_key in metric_data[key]:
                    scoring_data = metric_data[key][scoring_key]
                    break
        
        if scoring_data is None:
            available = self._find_available_scoring(hazard_code, metric)
            raise ValueError(
                f"No '{scoring_key}' scoring found for {hazard_code}.{metric}.\n"
                f"Available scoring: {available}"
            )
        
        # Extract thresholds and scores
        if isinstance(scoring_data, dict):
            thresholds = scoring_data.get('thresholds', [])
            scores = scoring_data.get('scores', [])
        elif isinstance(scoring_data, list):
            # Handle case where scoring_data might be just a list of thresholds
            # This assumes scores are 1..n
            thresholds = scoring_data
            scores = list(range(1, len(thresholds)))
        else:
            raise ValueError(f"Unexpected scoring data format for {hazard_code}.{metric}")
        
        return thresholds, scores
    
    def _find_available_scoring(self, hazard_code: str, metric: str) -> List[str]:
        """Find all available scoring systems for a hazard metric."""
        hazard = self.config['hazards'][hazard_code]
        metric_data = hazard['metrics'][metric]
        
        available = []
        
        def extract_scoring_keys(obj, prefix=''):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.endswith('_point'):
                        available.append(f"{prefix}{key}")
                    elif isinstance(value, dict):
                        extract_scoring_keys(value, f"{prefix}{key}.")
        
        extract_scoring_keys(metric_data)
        return available
    
    def get_all_thresholds(self, hazard_code: str) -> Dict:
        """
        Get all available thresholds for a hazard.
        
        Args:
            hazard_code: e.g., 'ER', 'LS', 'CF'
        
        Returns:
            Dictionary with all scoring systems and their thresholds
        """
        if hazard_code not in self.config['hazards']:
            raise ValueError(f"Hazard '{hazard_code}' not found")
        
        hazard = self.config['hazards'][hazard_code]
        result = {
            'hazard': hazard.get('name', hazard_code),
            'code': hazard_code,
            'metrics': {}
        }
        
        for metric_name, metric_data in hazard['metrics'].items():
            result['metrics'][metric_name] = {}
            
            def extract_scoring(obj, current_path):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key.endswith('_point'):
                            if isinstance(value, dict):
                                result['metrics'][metric_name][f"{current_path}{key}"] = {
                                    'thresholds': value.get('thresholds', []),
                                    'scores': value.get('scores', [])
                                }
                            elif isinstance(value, list):
                                result['metrics'][metric_name][f"{current_path}{key}"] = {
                                    'thresholds': value,
                                    'scores': list(range(1, len(value) + 1))
                                }
                        else:
                            extract_scoring(value, f"{current_path}{key}.")
            
            extract_scoring(metric_data, '')
        
        return result
    
    def score_value(self, 
                   hazard_code: str, 
                   value: float, 
                   scoring: Union[str, int] = '5',
                   metric: Optional[str] = None,
                   threshold_type: Optional[str] = None) -> int:
        """
        Score a value for a specific hazard.
        
        Args:
            hazard_code: e.g., 'ER', 'LS', 'CF'
            value: The value to score
            scoring: '5', '10', '3', '100' or '5_point', '10_point', etc.
            metric: Metric name (if None, uses the first available metric)
            threshold_type: For hazards with multiple thresholds
        
        Returns:
            Score (integer)
        
        Example:
            score = config.score_value('ER', 15, '5')  # Returns 3
            score = config.score_value('CF', 100, '10')  # Returns 6
        """
        thresholds, scores = self.get_thresholds(hazard_code, scoring, metric, threshold_type)
        
        # Check if thresholds are descending (like for return periods)
        is_descending = len(thresholds) > 1 and thresholds[0] > thresholds[-1]
        
        if is_descending:
            # For descending thresholds, reverse for digitize
            thresholds_asc = list(reversed(thresholds))
            scores_asc = list(reversed(scores))
            bin_idx = np.digitize(value, thresholds_asc) - 1
        else:
            # Normal ascending thresholds
            bin_idx = np.digitize(value, thresholds) - 1
        
        # Handle edge cases
        bin_idx = max(0, min(bin_idx, len(scores) - 1))
        
        return int(scores[bin_idx])
    
    def score_multiple(self, 
                      hazard_code: str, 
                      values: List[float], 
                      scoring: Union[str, int] = '5',
                      metric: Optional[str] = None,
                      threshold_type: Optional[str] = None) -> List[int]:
        """
        Score multiple values for a specific hazard.
        
        Args:
            hazard_code: e.g., 'ER', 'LS', 'CF'
            values: List of values to score
            scoring: '5', '10', '3', '100' or '5_point', '10_point', etc.
            metric: Metric name (if None, uses the first available metric)
            threshold_type: For hazards with multiple thresholds
        
        Returns:
            List of scores
        """
        return [self.score_value(hazard_code, v, scoring, metric, threshold_type) for v in values]
    
    def get_metadata(self) -> Dict:
        """Get metadata from the configuration."""
        return self.config.get('metadata', {})
    
    def __repr__(self) -> str:
        return f"ScoringConfig({self.config_path}, hazards={len(self.list_hazards())})"


# Create convenience functions
_config_instance = None

def get_config(config_path: Optional[Union[str, Path]] = None) -> ScoringConfig:
    """
    Get or create a ScoringConfig instance (singleton pattern).
    
    Args:
        config_path: Path to config file (optional)
    
    Returns:
        ScoringConfig instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ScoringConfig(config_path)
    return _config_instance


def get_thresholds(hazard_code: str, scoring: Union[str, int] = '5', **kwargs) -> Tuple[List[float], List[int]]:
    """
    Quick function to get thresholds.
    
    Args:
        hazard_code: e.g., 'ER', 'LS', 'CF'
        scoring: '5', '10', '3', '100' or '5_point', '10_point', etc.
        **kwargs: Additional arguments passed to ScoringConfig.get_thresholds
    
    Returns:
        Tuple of (thresholds list, scores list)
    """
    return get_config().get_thresholds(hazard_code, scoring, **kwargs)


def score(hazard_code: str, value: float, scoring: Union[str, int] = '5', **kwargs) -> int:
    """
    Quick function to score a value.
    
    Args:
        hazard_code: e.g., 'ER', 'LS', 'CF'
        value: The value to score
        scoring: '5', '10', '3', '100' or '5_point', '10_point', etc.
        **kwargs: Additional arguments passed to ScoringConfig.score_value
    
    Returns:
        Score (integer)
    """
    return get_config().score_value(hazard_code, value, scoring, **kwargs)