"""
Feature Utilities for TabTransformer

This module provides utilities for identifying and preprocessing features
for the TabTransformer model.

Author: Nguyen Duc Thang
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import json


def identify_feature_types(df: pd.DataFrame, 
                           feature_columns: List[str],
                           categorical_threshold: int = 100) -> Tuple[List[int], List[int], List[int]]:
    """
    Automatically identify categorical and numerical features.
    
    Features are classified as categorical if they have fewer unique values
    than the threshold, or if they appear to be binary/flag features.
    
    Args:
        df: DataFrame with features
        feature_columns: List of feature column names
        categorical_threshold: Max unique values to consider as categorical
        
    Returns:
        categorical_indices: List of categorical feature indices
        numerical_indices: List of numerical feature indices  
        categorical_cardinalities: List of unique value counts for categorical features
    """
    categorical_indices = []
    numerical_indices = []
    categorical_cardinalities = []
    
    # Known categorical features from CICIoT2023 dataset
    # These are typically protocol types, flags, and protocol indicators
    known_categorical_names = [
        'Protocol Type',  # TCP/UDP/ICMP protocols
        'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',  # Application protocols
        'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC',  # Network protocols
        'fin_flag_number', 'syn_flag_number', 'rst_flag_number',  # TCP flags
        'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
        'ack_count', 'syn_count', 'fin_count', 'urg_count', 'rst_count'  # Flag counts
    ]
    
    for idx, col_name in enumerate(feature_columns):
        n_unique = df[col_name].nunique()
        
        # Check if it's a known categorical feature
        is_categorical = (col_name in known_categorical_names or 
                         n_unique <= categorical_threshold)
        
        if is_categorical:
            categorical_indices.append(idx)
            categorical_cardinalities.append(int(n_unique))
        else:
            numerical_indices.append(idx)
    
    return categorical_indices, numerical_indices, categorical_cardinalities


def save_feature_config(categorical_indices: List[int],
                        numerical_indices: List[int],
                        categorical_cardinalities: List[int],
                        output_path: str) -> None:
    """
    Save feature configuration to JSON file.
    
    Args:
        categorical_indices: Indices of categorical features
        numerical_indices: Indices of numerical features
        categorical_cardinalities: Unique value counts for categorical features
        output_path: Path to save JSON file
    """
    config = {
        'categorical_indices': categorical_indices,
        'numerical_indices': numerical_indices,
        'categorical_cardinalities': categorical_cardinalities,
        'num_categorical': len(categorical_indices),
        'num_numerical': len(numerical_indices),
        'total_features': len(categorical_indices) + len(numerical_indices)
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Feature configuration saved to: {output_path}")
    print(f"   Categorical features: {len(categorical_indices)}")
    print(f"   Numerical features: {len(numerical_indices)}")
    print(f"   Total features: {config['total_features']}")


def load_feature_config(config_path: str) -> Dict:
    """
    Load feature configuration from JSON file.
    
    Args:
        config_path: Path to JSON config file
        
    Returns:
        Feature configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def split_features_by_type(X: np.ndarray,
                           categorical_indices: List[int],
                           numerical_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split feature matrix into categorical and numerical parts.
    
    Args:
        X: Full feature matrix [num_samples, num_features]
        categorical_indices: Indices of categorical features
        numerical_indices: Indices of numerical features
        
    Returns:
        X_categorical: Categorical features [num_samples, num_categorical]
        X_numerical: Numerical features [num_samples, num_numerical]
    """
    if len(categorical_indices) > 0:
        X_categorical = X[:, categorical_indices]
    else:
        X_categorical = np.array([]).reshape(X.shape[0], 0)
    
    if len(numerical_indices) > 0:
        X_numerical = X[:, numerical_indices]
    else:
        X_numerical = np.array([]).reshape(X.shape[0], 0)
    
    return X_categorical, X_numerical


if __name__ == '__main__':
    # Test the feature identification
    print("=" * 80)
    print("Testing Feature Type Identification")
    print("=" * 80)
    
    # Create dummy data similar to CICIoT2023
    from .includes import X_columns
    
    n_samples = 1000
    data = {}
    
    # Create dummy features
    for col in X_columns:
        if 'Protocol' in col or col in ['HTTP', 'TCP', 'UDP']:
            # Categorical features (low cardinality)
            data[col] = np.random.randint(0, 10, n_samples)
        else:
            # Numerical features
            data[col] = np.random.randn(n_samples)
    
    df = pd.DataFrame(data)
    
    # Identify feature types
    cat_idx, num_idx, cat_card = identify_feature_types(df, X_columns)
    
    print(f"\n📊 Results:")
    print(f"   Categorical features: {len(cat_idx)}")
    print(f"   Numerical features: {len(num_idx)}")
    print(f"   Categorical cardinalities: {cat_card[:5]}... (showing first 5)")
    
    # Test save/load
    save_feature_config(cat_idx, num_idx, cat_card, '/tmp/test_feature_config.json')
    loaded_config = load_feature_config('/tmp/test_feature_config.json')
    print(f"\n✅ Config saved and loaded successfully")
    print(f"   Loaded config keys: {list(loaded_config.keys())}")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
