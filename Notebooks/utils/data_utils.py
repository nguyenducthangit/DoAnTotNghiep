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

# Import from local includes module
from .includes import X_columns, y_column, dict_34_classes


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


def encode_labels(df_train: pd.DataFrame, label_col: str = y_column, 
                  save_path: str = None) -> Tuple[pd.DataFrame, LabelEncoder, Dict]:
    """
    Encode attack labels from strings to numeric values (0-33).
    
    Args:
        df_train: DataFrame with label column
        label_col: Name of the label column
        save_path: Path to save the label encoder (optional)
        
    Returns:
        Tuple of (DataFrame with encoded labels, LabelEncoder, label mapping dict)
    """
    print(f"\n🏷️  Encoding labels...")
    
    # Check label distribution
    label_counts = df_train[label_col].value_counts()
    print(f"   Found {len(label_counts)} unique labels:")
    for label, count in label_counts.head(10).items():
        print(f"      {label}: {count:,} ({count/len(df_train)*100:.2f}%)")
    if len(label_counts) > 10:
        print(f"      ... and {len(label_counts) - 10} more")
    
    # Create label encoder
    le = LabelEncoder()
    df_train[label_col] = le.fit_transform(df_train[label_col])
    
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
    
    return df_train, le, label_mapping


def normalize_features(df_train: pd.DataFrame, feature_cols: List[str] = X_columns,
                       save_path: str = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Normalize features using MinMaxScaler to [0, 1] range.
    
    Args:
        df_train: DataFrame with features
        feature_cols: List of feature column names
        save_path: Path to save the scaler (optional)
        
    Returns:
        Tuple of (DataFrame with normalized features, fitted scaler)
    """
    print(f"\n📏 Normalizing features...")
    
    # Check if all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df_train.columns]
    if missing_cols:
        print(f"   ⚠️  Warning: {len(missing_cols)} feature columns not found in dataset")
        print(f"      Missing: {missing_cols[:5]}...")
        feature_cols = [col for col in feature_cols if col in df_train.columns]
        print(f"   Using {len(feature_cols)} available features")
    
    # Fit scaler
    scaler = MinMaxScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    
    print(f"   ✓ Normalized {len(feature_cols)} features to [0, 1] range")
    
    # Save scaler
    if save_path:
        scaler_path = os.path.join(save_path, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"   💾 Saved scaler to: {scaler_path}")
    
    return df_train, scaler


def partition_data_noniid(df_train: pd.DataFrame, num_clients: int = 5, 
                          label_col: str = y_column,
                          test_split: float = 0.2,
                          random_seed: int = 42) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Partition data into Non-IID distributions for federated learning clients.
    Each client gets a majority of specific attack types to simulate heterogeneity.
    
    Args:
        df_train: DataFrame with features and labels (TRAINING DATA ONLY, test already separated)
        num_clients: Number of clients to partition data for
        label_col: Name of the label column
        test_split: Fraction of data to reserve for testing (IGNORED - for backward compatibility)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with client data: {'client_0': {'X': ..., 'y': ...}}
        Note: 'test' should be added separately after calling this function
    """
    print(f"\n🔀 Partitioning data for {num_clients} clients (Non-IID)...")
    
    # Separate features and labels
    X_train = df_train.drop(columns=[label_col]).values
    y_train = df_train[label_col].values
    
    print(f"   Total training samples: {len(X_train):,}")
    
    # Check what labels actually exist in the data
    unique_labels = np.unique(y_train)
    print(f"   Found {len(unique_labels)} unique labels in training data")
    
    # Count samples per label
    from collections import Counter
    label_counts = Counter(y_train)
    print(f"\n   Label distribution (showing labels with < 100 samples):")
    for label in sorted(unique_labels):
        count = label_counts[label]
        if count < 100:
            print(f"      Label {label}: {count} samples")
    
    # Define attack type groups for Non-IID distribution
    # IMPORTANT: Ensure ALL labels (0-33) are covered
    attack_groups = {
        'DDoS': list(range(4, 16)),      # DDoS attacks (labels 4-15)
        'DoS': list(range(16, 20)),      # DoS attacks (labels 16-19)
        'Mirai': list(range(20, 23)),    # Mirai attacks (labels 20-22)
        'Recon': list(range(23, 28)),    # Reconnaissance (labels 23-27)
        'Others': [0, 1, 2, 3, 28, 29, 30, 31, 32, 33]  # All other attack types
    }
    
    # Verify all labels are covered
    all_grouped_labels = set()
    for group_labels in attack_groups.values():
        all_grouped_labels.update(group_labels)
    
    # Find any labels not in groups
    ungrouped_labels = set(unique_labels) - all_grouped_labels
    if ungrouped_labels:
        print(f"\n   ⚠️  Found {len(ungrouped_labels)} labels not in attack groups: {sorted(ungrouped_labels)}")
        print(f"   → Adding to 'Others' group")
        attack_groups['Others'].extend(sorted(ungrouped_labels))
    
    # Verify all existing labels are now in groups
    all_grouped_labels = set()
    for group_labels in attack_groups.values():
        all_grouped_labels.update(group_labels)
    
    missing_labels = set(unique_labels) - all_grouped_labels
    if missing_labels:
        raise ValueError(f"ERROR: Labels {sorted(missing_labels)} are not assigned to any group!")
    
    print(f"\n   ✅ All {len(unique_labels)} labels are covered in attack groups")
    
    # Partition strategy: Each client gets majority (70%) of specific attack group
    # Remaining 30% goes to other clients for some overlap
    client_data = {}
    assigned_indices = set()
    
    # Assign attack groups to clients
    group_names = list(attack_groups.keys())[:num_clients]  # Take first N groups
    
    for client_id in range(num_clients):
        if client_id < len(group_names):
            # Assign majority of specific attack type
            group_name = group_names[client_id]
            group_labels = attack_groups[group_name]
            
            # Find all indices for this attack group
            group_indices = [i for i in range(len(X_train)) 
                           if y_train[i] in group_labels]
            
            if len(group_indices) > 0:
                # Take 70% of this group
                np.random.seed(random_seed + client_id)
                n_samples = max(1, int(len(group_indices) * 0.7))
                selected = np.random.choice(group_indices, size=n_samples, replace=False)
                
                # Mark as assigned
                assigned_indices.update(selected)
                
                client_data[f'client_{client_id}'] = {
                    'X': X_train[selected],
                    'y': y_train[selected]
                }
                
                print(f"   Client {client_id} ({group_name}): {len(selected):,} samples from labels {group_labels[:5]}{'...' if len(group_labels) > 5 else ''}")
            else:
                print(f"   ⚠️  Client {client_id} ({group_name}): No samples found for this group!")
                client_data[f'client_{client_id}'] = {
                    'X': np.array([]).reshape(0, X_train.shape[1]),
                    'y': np.array([], dtype=y_train.dtype)
                }
    
    # Distribute ALL remaining unassigned data across clients
    remaining_unassigned = [i for i in range(len(X_train)) if i not in assigned_indices]
    
    if remaining_unassigned:
        print(f"\n   Distributing {len(remaining_unassigned):,} remaining samples across all clients...")
        
        # Split remaining data evenly across all clients
        np.random.seed(random_seed)
        np.random.shuffle(remaining_unassigned)
        
        chunk_size = len(remaining_unassigned) // num_clients
        
        for client_id in range(num_clients):
            # Get chunk for this client
            start_idx = client_id * chunk_size
            end_idx = (client_id + 1) * chunk_size if client_id < num_clients - 1 else len(remaining_unassigned)
            client_remaining = remaining_unassigned[start_idx:end_idx]
            
            if len(client_remaining) > 0:
                # Append to client
                if len(client_data[f'client_{client_id}']['X']) > 0:
                    client_data[f'client_{client_id}']['X'] = np.vstack([
                        client_data[f'client_{client_id}']['X'],
                        X_train[client_remaining]
                    ])
                    client_data[f'client_{client_id}']['y'] = np.concatenate([
                        client_data[f'client_{client_id}']['y'],
                        y_train[client_remaining]
                    ])
                else:
                    client_data[f'client_{client_id}'] = {
                        'X': X_train[client_remaining],
                        'y': y_train[client_remaining]
                    }
                
                print(f"      → Client {client_id} gets +{len(client_remaining):,} samples")
    
    # Verify all data is assigned
    total_assigned = sum(len(data['X']) for data in client_data.values())
    
    print(f"\n   📊 Final distribution:")
    for client_id in range(num_clients):
        n_samples = len(client_data[f'client_{client_id}']['X'])
        pct = (n_samples / len(X_train)) * 100 if len(X_train) > 0 else 0
        print(f"      Client {client_id}: {n_samples:,} samples ({pct:.1f}%)")
    
    if total_assigned != len(X_train):
        print(f"\n   ❌ ERROR: {len(X_train) - total_assigned:,} samples not assigned!")
        raise ValueError(f"Data assignment error: {total_assigned} assigned vs {len(X_train)} total")
    else:
        print(f"\n   ✅ All {len(X_train):,} training samples assigned successfully")
    
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


def filter_features_by_names(
    df: pd.DataFrame,
    selected_features: List[str],
    label_col: str = y_column
) -> pd.DataFrame:
    """
    Filter DataFrame to include only selected features and label column.
    
    Args:
        df: DataFrame with all features
        selected_features: List of feature names to keep
        label_col: Name of label column
        
    Returns:
        Filtered DataFrame
    """
    # Ensure label column is included
    cols_to_keep = list(selected_features) + [label_col]
    
    # Check if all selected features exist
    missing_features = set(selected_features) - set(df.columns)
    if missing_features:
        raise ValueError(f"Features not found in DataFrame: {missing_features}")
    
    # Filter
    df_filtered = df[cols_to_keep].copy()
    
    print(f"✅ Filtered features: {len(selected_features)} features retained")
    return df_filtered


def load_selected_features_list(json_path: str) -> List[str]:
    """
    Load selected feature list from GSA results JSON.
    
    Args:
        json_path: Path to selected_features.json from GSA
        
    Returns:
        List of selected feature names
    """
    import json
    
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    selected_features = results['selected_features']
    print(f"✅ Loaded {len(selected_features)} selected features from {json_path}")
    
    return selected_features


def save_feature_list(feature_names: List[str], save_path: str) -> None:
    """
    Save feature list to JSON file.
    
    Args:
        feature_names: List of feature names
        save_path: Path to save JSON file
    """
    import json
    
    feature_data = {
        'features': feature_names,
        'num_features': len(feature_names),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(save_path, 'w') as f:
        json.dump(feature_data, f, indent=2)
    
    print(f"✅ Saved {len(feature_names)} features to {save_path}")


if __name__ == '__main__':
    # Test the functions
    print("=" * 80)
    print("Testing Data Utilities")
    print("=" * 80)
    
    # This is just for testing - actual usage will be in notebooks
    print("\n✅ All functions defined successfully!")
    print("   Use these functions in 1_Data_Preprocessing.ipynb")
