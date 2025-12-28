"""
Data Utilities for Federated Learning IoT Attack Detection

This module provides functions for loading, cleaning, preprocessing, and partitioning
the CICIoT2023 dataset for federated learning training.

Author: Nguyen Duc Thang
Project: IoT Network Attack Detection using Federated Learning
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import sys

# Add parent directory to path to import includes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from includes import X_columns, y_column, dict_34_classes


def load_dataset_chunked(data_dir: str, chunk_size: int = 50000, 
                         sample_fraction: float = None) -> pd.DataFrame:
    """
    Load all CSV files from the dataset directory using chunking to avoid memory overflow.
    
    Args:
        data_dir: Path to directory containing CSV files
        chunk_size: Number of rows to read at a time
        sample_fraction: If provided, randomly sample this fraction of data (for testing)
        
    Returns:
        Combined DataFrame with all data
    """
    print(f"📂 Loading dataset from: {data_dir}")
    
    # Get all CSV files
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    print(f"   Found {len(csv_files)} CSV files")
    
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    all_chunks = []
    total_rows = 0
    
    for i, csv_file in enumerate(csv_files):
        file_path = os.path.join(data_dir, csv_file)
        print(f"   [{i+1}/{len(csv_files)}] Loading {csv_file}...", end=" ")
        
        try:
            # Read file in chunks
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
            
            # Combine chunks from this file
            df_file = pd.concat(chunks, ignore_index=True)
            rows = len(df_file)
            total_rows += rows
            print(f"✓ {rows:,} rows")
            
            all_chunks.append(df_file)
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    # Combine all files
    print(f"\n🔗 Combining all files...")
    df = pd.concat(all_chunks, ignore_index=True)
    print(f"   Total rows: {len(df):,}")
    print(f"   Total columns: {len(df.columns)}")
    
    # Sample if requested
    if sample_fraction is not None and 0 < sample_fraction < 1:
        print(f"\n🎲 Sampling {sample_fraction*100}% of data for testing...")
        df = df.sample(frac=sample_fraction, random_state=42)
        print(f"   Sampled rows: {len(df):,}")
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and removing duplicates.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    print(f"\n🧹 Cleaning data...")
    initial_rows = len(df)
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"   Missing values found:")
        for col in missing[missing > 0].index:
            print(f"      {col}: {missing[col]} ({missing[col]/len(df)*100:.2f}%)")
        
        # Drop rows with missing values (or you can use imputation)
        df = df.dropna()
        print(f"   Dropped {initial_rows - len(df):,} rows with missing values")
    else:
        print(f"   ✓ No missing values")
    
    # Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"   Removed {duplicates:,} duplicate rows")
    else:
        print(f"   ✓ No duplicates")
    
    print(f"   Final rows: {len(df):,}")
    
    return df


def encode_labels(df: pd.DataFrame, label_col: str = y_column, 
                  save_path: str = None) -> Tuple[pd.DataFrame, LabelEncoder, Dict]:
    """
    Encode attack labels from strings to numeric values (0-33).
    
    Args:
        df: DataFrame with label column
        label_col: Name of the label column
        save_path: Path to save the label encoder (optional)
        
    Returns:
        Tuple of (DataFrame with encoded labels, LabelEncoder, label mapping dict)
    """
    print(f"\n🏷️  Encoding labels...")
    
    # Check label distribution
    label_counts = df[label_col].value_counts()
    print(f"   Found {len(label_counts)} unique labels:")
    for label, count in label_counts.head(10).items():
        print(f"      {label}: {count:,} ({count/len(df)*100:.2f}%)")
    if len(label_counts) > 10:
        print(f"      ... and {len(label_counts) - 10} more")
    
    # Create label encoder
    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col])
    
    # Create human-readable mapping
    label_mapping = {i: label for i, label in enumerate(le.classes_)}
    
    print(f"   ✓ Encoded {len(le.classes_)} classes to numeric values (0-{len(le.classes_)-1})")
    
    # Save encoder and mapping
    if save_path:
        # Save encoder
        encoder_path = os.path.join(save_path, 'label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(le, f)
        print(f"   💾 Saved label encoder to: {encoder_path}")
        
        # Save JSON mapping
        json_path = os.path.join(save_path, 'labels.json')
        with open(json_path, 'w') as f:
            json.dump(label_mapping, f, indent=2)
        print(f"   💾 Saved label mapping to: {json_path}")
    
    return df, le, label_mapping


def normalize_features(df: pd.DataFrame, feature_cols: List[str] = X_columns,
                       save_path: str = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Normalize features using MinMaxScaler to [0, 1] range.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        save_path: Path to save the scaler (optional)
        
    Returns:
        Tuple of (DataFrame with normalized features, fitted scaler)
    """
    print(f"\n📏 Normalizing features...")
    
    # Check if all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"   ⚠️  Warning: {len(missing_cols)} feature columns not found in dataset")
        print(f"      Missing: {missing_cols[:5]}...")
        feature_cols = [col for col in feature_cols if col in df.columns]
        print(f"   Using {len(feature_cols)} available features")
    
    # Fit scaler
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    print(f"   ✓ Normalized {len(feature_cols)} features to [0, 1] range")
    
    # Save scaler
    if save_path:
        scaler_path = os.path.join(save_path, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"   💾 Saved scaler to: {scaler_path}")
    
    return df, scaler


def partition_data_noniid(df: pd.DataFrame, num_clients: int = 5, 
                          label_col: str = y_column,
                          test_split: float = 0.2,
                          random_seed: int = 42) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Partition data into Non-IID distributions for federated learning clients.
    Each client gets a majority of specific attack types to simulate heterogeneity.
    
    Args:
        df: DataFrame with features and labels
        num_clients: Number of clients to partition data for
        label_col: Name of the label column
        test_split: Fraction of data to reserve for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with client data: {'client_0': {'X': ..., 'y': ...}, 'test': {...}}
    """
    print(f"\n🔀 Partitioning data for {num_clients} clients (Non-IID)...")
    
    # Separate features and labels
    X = df.drop(columns=[label_col]).values
    y = df[label_col].values
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_seed, stratify=y
    )
    
    print(f"   Train set: {len(X_train):,} samples")
    print(f"   Test set: {len(X_test):,} samples")
    
    # Define attack type groups (based on dict_34_classes from includes.py)
    attack_groups = {
        'DDoS': list(range(1, 13)),      # DDoS attacks: 1-12
        'DoS': list(range(13, 17)),      # DoS attacks: 13-16
        'Mirai': list(range(17, 20)),    # Mirai: 17-19
        'Recon': list(range(20, 25)),    # Reconnaissance: 20-24
        'Spoofing': list(range(25, 27)), # Spoofing: 25-26
        'Web': list(range(27, 33)),      # Web attacks: 27-32
        'BruteForce': [33]               # Brute Force: 33
    }
    
    # Partition strategy: Each client gets 70% of a specific attack group
    client_data = {}
    remaining_indices = list(range(len(X_train)))
    
    for client_id in range(num_clients):
        if client_id < len(attack_groups):
            # Assign majority of specific attack type
            group_name = list(attack_groups.keys())[client_id]
            group_labels = attack_groups[group_name]
            
            # Find indices for this attack group
            group_indices = [i for i in remaining_indices 
                           if y_train[i] in group_labels]
            
            # Take 70% of this group
            np.random.seed(random_seed + client_id)
            n_samples = int(len(group_indices) * 0.7)
            selected = np.random.choice(group_indices, size=n_samples, replace=False)
            
            # Remove from remaining
            remaining_indices = [i for i in remaining_indices if i not in selected]
            
            client_data[f'client_{client_id}'] = {
                'X': X_train[selected],
                'y': y_train[selected]
            }
            
            print(f"   Client {client_id} ({group_name}): {len(selected):,} samples")
            
        else:
            # Last client gets mixed distribution from remaining data
            client_data[f'client_{client_id}'] = {
                'X': X_train[remaining_indices],
                'y': y_train[remaining_indices]
            }
            print(f"   Client {client_id} (Mixed): {len(remaining_indices):,} samples")
    
    # Add test set
    client_data['test'] = {
        'X': X_test,
        'y': y_test
    }
    
    return client_data


def save_partitioned_data(client_data: Dict, save_dir: str):
    """
    Save partitioned client data as .npz files.
    
    Args:
        client_data: Dictionary with client data
        save_dir: Directory to save the data files
    """
    print(f"\n💾 Saving partitioned data to: {save_dir}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    for client_name, data in client_data.items():
        file_path = os.path.join(save_dir, f'{client_name}_data.npz')
        np.savez_compressed(file_path, X=data['X'], y=data['y'])
        print(f"   ✓ Saved {client_name}: {file_path} ({len(data['X']):,} samples)")
    
    print(f"   ✅ All data saved successfully!")


# Utility function to load saved data
def load_client_data(data_dir: str, client_name: str) -> Dict[str, np.ndarray]:
    """
    Load client data from .npz file.
    
    Args:
        data_dir: Directory containing data files
        client_name: Name of the client (e.g., 'client_0', 'test')
        
    Returns:
        Dictionary with 'X' and 'y' arrays
    """
    file_path = os.path.join(data_dir, f'{client_name}_data.npz')
    data = np.load(file_path)
    return {'X': data['X'], 'y': data['y']}


if __name__ == '__main__':
    # Test the functions
    print("=" * 80)
    print("Testing Data Utilities")
    print("=" * 80)
    
    # This is just for testing - actual usage will be in notebooks
    print("\n✅ All functions defined successfully!")
    print("   Use these functions in 1_Data_Preprocessing.ipynb")
