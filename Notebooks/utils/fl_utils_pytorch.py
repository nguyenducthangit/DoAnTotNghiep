
"""
Federated Learning Utilities for PyTorch

This module provides FL training functions adapted for PyTorch models,
including client training, server aggregation (FedAvg), and evaluation.

Author: Nguyen Duc Thang
Project: IoT Network Attack Detection using Federated Learning
Framework: PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import OrderedDict
import copy


def create_data_loaders(X: np.ndarray, 
                        y:np.ndarray,
                        batch_size: int = 256,
                        shuffle: bool = True) -> DataLoader:
    """
    Create PyTorch DataLoader from numpy arrays.
    
    Args:
        X: Input features [num_samples, num_features]
        y: Labels [num_samples]
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        
    Returns:
        PyTorch DataLoader
    """
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader


def split_features(X: np.ndarray,
                   num_categorical: int = 20,
                   categorical_cardinalities: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split feature matrix into categorical and numerical features.
    
    Since data is normalized (MinMaxScaler), categorical features are in [0, 1] range.
    We need to scale them back to [0, cardinality-1] for embedding lookup.
    
    Args:
        X: Full feature matrix [num_samples, total_features] (normalized to [0, 1])
        num_categorical: Number of categorical features (first N columns)
        categorical_cardinalities: List of unique value counts for each categorical feature
        
    Returns:
        categorical_features: [num_samples, num_categorical] (integer indices for embedding)
        numerical_features: [num_samples, num_numerical] (normalized float values)
    """
    categorical = X[:, :num_categorical].copy()
    numerical = X[:, num_categorical:]
    
    # Scale categorical features from [0, 1] to [0, cardinality-1]
    if categorical_cardinalities is not None and len(categorical_cardinalities) == num_categorical:
        for i in range(num_categorical):
            cardinality = categorical_cardinalities[i]
            # Scale from [0, 1] to [0, cardinality-1]
            # Use floor to ensure values are in [0, cardinality-1]
            categorical[:, i] = np.clip(
                np.floor(categorical[:, i] * cardinality).astype(np.int64),
                0, cardinality - 1
            )
    else:
        # Fallback: assume binary or small cardinality features
        # Scale to [0, 99] as default max
        categorical = np.clip(
            np.floor(categorical * 100).astype(np.int64),
            0, 99
        )
    
    return categorical, numerical


def train_client_pytorch(model: nn.Module,
                        train_loader: DataLoader,
                        epochs: int = 5,
                        learning_rate: float = 0.001,
                        device: str = 'cpu',
                        num_categorical: int = 20,
                        categorical_cardinalities: Optional[List[int]] = None,
                        verbose: bool = False) -> Tuple[OrderedDict, float, float]:
    """
    Train model locally on client data (PyTorch).
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        epochs: Number of local training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        num_categorical: Number of categorical features for splitting
        categorical_cardinalities: List of unique value counts for each categorical feature
        verbose: Whether to print training progress
        
    Returns:
        state_dict: Updated model weights
        avg_loss: Average training loss
        avg_accuracy: Average training accuracy
    """
    model.to(device)
    model.train()
    
    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Split features
            cat_features, num_features = split_features(
                data.cpu().numpy(), 
                num_categorical,
                categorical_cardinalities
            )
            cat_features = torch.LongTensor(cat_features).to(device)
            num_features = torch.FloatTensor(num_features).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(cat_features, num_features)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            epoch_correct += pred.eq(target).sum().item()
            epoch_samples += data.size(0)
        
        # Epoch metrics
        epoch_loss /= epoch_samples
        epoch_accuracy = epoch_correct / epoch_samples
        
        total_loss += epoch_loss
        total_correct += epoch_correct
        total_samples += epoch_samples
        
        if verbose:
            print(f"   Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_accuracy:.4f}")
    
    # Average metrics across all epochs
    avg_loss = total_loss / epochs
    avg_accuracy = total_correct / total_samples
    
    return model.state_dict(), avg_loss, avg_accuracy


def federated_averaging_pytorch(client_state_dicts: List[OrderedDict],
                                client_weights: List[int]) -> OrderedDict:
    """
    Aggregate client model weights using Federated Averaging (FedAvg).
    
    Args:
        client_state_dicts: List of state_dict from each client
        client_weights: Number of samples per client (for weighted averaging)
        
    Returns:
        Aggregated state_dict
    """
    if len(client_state_dicts) == 0:
        raise ValueError("No client models to aggregate")
    
    total_weight = sum(client_weights)
    
    # Initialize aggregated state dict with zeros
    aggregated_state_dict = OrderedDict()
    
    # Get structure from first client
    for key in client_state_dicts[0].keys():
        aggregated_state_dict[key] = torch.zeros_like(client_state_dicts[0][key], dtype=torch.float32)
    
    # Weighted averaging
    for client_idx, state_dict in enumerate(client_state_dicts):
        weight = client_weights[client_idx] / total_weight
        
        for key in state_dict.keys():
            aggregated_state_dict[key] += weight * state_dict[key].float()
    
    return aggregated_state_dict


def evaluate_model_pytorch(model: nn.Module,
                          test_loader: DataLoader,
                          device: str = 'cpu',
                          num_categorical: int = 20,
                          categorical_cardinalities: Optional[List[int]] = None,
                          verbose: bool = True) -> Tuple[float, float]:
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to run evaluation on
        num_categorical: Number of categorical features
        categorical_cardinalities: List of unique value counts for each categorical feature
        verbose: Whether to print results
        
    Returns:
        test_loss: Average test loss
        test_accuracy: Test accuracy
    """
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Split features
            cat_features, num_features = split_features(
                data.cpu().numpy(), 
                num_categorical,
                categorical_cardinalities
            )
            cat_features = torch.LongTensor(cat_features).to(device)
            num_features = torch.FloatTensor(num_features).to(device)
            
            # Forward pass
            output = model(cat_features, num_features)
            loss = criterion(output, target)
            
            # Track metrics
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()
            total_samples += data.size(0)
    
    test_loss = total_loss / total_samples
    test_accuracy = total_correct / total_samples
    
    if verbose:
        print(f"\n📊 Test Set Results:")
        print(f"   Loss: {test_loss:.4f}")
        print(f"   Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    return test_loss, test_accuracy


def federated_training_loop_pytorch(
        global_model: nn.Module,
        client_data_loaders: List[DataLoader],
        test_loader: DataLoader,
        num_rounds: int = 30,
        local_epochs: int = 5,
        learning_rate: float = 0.001,
        device: str = 'cpu',
        num_categorical: int = 20,
        categorical_cardinalities: Optional[List[int]] = None,
        verbose: bool = True,
        aggregation_method: str = 'fedavg',
        aggregation_config: Optional[Dict] = None,
        client_metrics_history: Optional[List] = None) -> Dict:
    """
    Main federated learning training loop with PyTorch.
    
    Args:
        global_model: Initial global model
        client_data_loaders: List of DataLoaders, one per client
        test_loader: Test data loader
        num_rounds: Number of FL communication rounds
        local_epochs: Local training epochs per round
        learning_rate: Learning rate for client training
        device: Device for training
        num_categorical: Number of categorical features
        verbose: Verbose output
        
    Returns:
        history: Dictionary with training history
    """
    num_clients = len(client_data_loaders)
    
    # Training history
    history = {
        'round': [],
        'loss': [],
        'accuracy': [],
        'client_losses': [],
        'client_accuracies': []
    }
    
    print(f"\n{'='*80}")
    print("FEDERATED LEARNING TRAINING (PyTorch + TabTransformer)")
    print(f"{'='*80}")
    print(f"Number of clients: {num_clients}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Local epochs per round: {local_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*80}")
        print(f"ROUND {round_num}/{num_rounds}")
        print(f"{'='*80}")
        
        client_state_dicts = []
        client_sizes = []
        client_losses = []
        client_accs = []
        client_metrics = []  # For FedMade
        
        # Client training
        print(f"📡 Distributing global model to {num_clients} clients...")
        
        for client_idx, train_loader in enumerate(client_data_loaders):
            if verbose:
                print(f"\n   Client {client_idx} training...")
            
            # Create local model copy
            local_model = copy.deepcopy(global_model)
            
            # Train locally
            state_dict, loss, acc = train_client_pytorch(
                model=local_model,
                train_loader=train_loader,
                epochs=local_epochs,
                learning_rate=learning_rate,
                device=device,
                num_categorical=num_categorical,
                categorical_cardinalities=categorical_cardinalities,
                verbose=False
            )
            
            client_state_dicts.append(state_dict)
            client_sizes.append(len(train_loader.dataset))
            client_losses.append(loss)
            client_accs.append(acc)
            
            # Collect metrics for FedMade
            if aggregation_method == 'fedmade':
                # Create validation set from client data (20% last)
                val_size = int(len(train_loader.dataset) * 0.2)
                if val_size > 0:
                    # Get validation data
                    val_indices = list(range(len(train_loader.dataset) - val_size, len(train_loader.dataset)))
                    val_subset = torch.utils.data.Subset(train_loader.dataset, val_indices)
                    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)
                    
                    # Evaluate on validation set
                    val_loss, val_acc = evaluate_model_pytorch(
                        model=local_model,
                        test_loader=val_loader,
                        device=device,
                        num_categorical=num_categorical,
                        categorical_cardinalities=categorical_cardinalities,
                        verbose=False
                    )
                else:
                    val_loss, val_acc = loss, acc
                
                client_metrics.append({
                    'accuracy': float(val_acc),
                    'loss': float(val_loss),
                    'num_samples': len(train_loader.dataset)
                })
                
                if verbose:
                    print(f"   ✓ Client {client_idx}: Loss={loss:.4f}, Acc={acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            else:
                if verbose:
                    print(f"   ✓ Client {client_idx}: Loss={loss:.4f}, Acc={acc:.4f}")
        
        # Server aggregation
        if aggregation_method == 'fedmade' and aggregation_config is not None:
            print(f"\n🤝 Aggregating với FedMade (Round {round_num})...")
            
            from utils.fedmade_aggregation import fedmade_aggregate_with_fallback
            
            aggregated_state_dict = fedmade_aggregate_with_fallback(
                client_state_dicts=client_state_dicts,
                client_metrics=client_metrics,
                client_weights=client_sizes,
                contribution_threshold=aggregation_config.get('contribution_threshold', 0.0),
                accuracy_weight=aggregation_config.get('accuracy_weight', 0.7),
                loss_weight=aggregation_config.get('loss_weight', 0.3),
                verbose=aggregation_config.get('verbose', False)
            )
            
            # Save contribution scores if history list provided
            if client_metrics_history is not None:
                round_scores = {
                    'round': round_num,
                    'clients': []
                }
                for i, metrics in enumerate(client_metrics):
                    round_scores['clients'].append({
                        'client_id': i,
                        'accuracy': float(metrics.get('accuracy', 0)),
                        'loss': float(metrics.get('loss', 0)),
                        'num_samples': int(metrics.get('num_samples', 0))
                    })
                client_metrics_history.append(round_scores)
            
            print(f"   ✓ Global model updated (FedMade)")
        else:
            print(f"\n🤝 Aggregating với FedAvg (Round {round_num})...")
            aggregated_state_dict = federated_averaging_pytorch(client_state_dicts, client_sizes)
            print(f"   ✓ Global model updated (FedAvg)")
            
            global_model.load_state_dict(aggregated_state_dict)
            
            # Evaluate global model
        print(f"\n📊 Evaluating global model on test set...")
        test_loss, test_accuracy = evaluate_model_pytorch(
            model=global_model,
            test_loader=test_loader,
            device=device,
            num_categorical=num_categorical,
            categorical_cardinalities=categorical_cardinalities,
            verbose=False
        )
        
        # Record history
        history['round'].append(round_num)
        history['loss'].append(test_loss)
        history['accuracy'].append(test_accuracy)
        history['client_losses'].append(client_losses)
        history['client_accuracies'].append(client_accs)
        
        # Print round summary
        print(f"\n{'-'*80}")
        print(f"ROUND {round_num} SUMMARY:")
        print(f"   Global Test Loss: {test_loss:.4f}")
        print(f"   Global Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"   Avg Client Loss: {np.mean(client_losses):.4f}")
        print(f"   Avg Client Accuracy: {np.mean(client_accs):.4f}")
        print(f"{'-'*80}")
        
        # Check if target achieved
        if test_accuracy >= 0.95:
            print(f"\n🎯 Target accuracy (≥95%) achieved at round {round_num}!")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED!")
    print(f"{'='*80}")
    print(f"Final Test Accuracy: {history['accuracy'][-1]*100:.2f}%")
    print(f"{'='*80}\n")
    
    return history


def save_checkpoint(model: nn.Module,
                   optimizer: Optional[optim.Optimizer],
                   round_num: int,
                   history: Dict,
                   filepath: str):
    """
    Save training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (optional)
        round_num: Current round number
        history: Training history
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'round': round_num,
        'model_state_dict': model.state_dict(),
        'history': history
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved to: {filepath}")


def load_checkpoint(filepath: str,
                   model: nn.Module,
                   optimizer: Optional[optim.Optimizer] = None,
                   device: str = 'cpu') -> Tuple[nn.Module, int, Dict]:
    """
    Load training checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        device: Device to load to
        
    Returns:
        model: Model with loaded weights
        round_num: Round number from checkpoint
        history: Training history from checkpoint
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    round_num = checkpoint['round']
    history = checkpoint['history']
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"📂 Checkpoint loaded from: {filepath}")
    print(f"   Round: {round_num}")
    print(f"   Accuracy: {history['accuracy'][-1]*100:.2f}%")
    
    return model, round_num, history


if __name__ == '__main__':
    # Test FL utilities
    print("=" * 80)
    print("Testing PyTorch FL Utilities")
    print("=" * 80)
    
    # Import TabTransformer for testing
    from model_utils_pytorch import TabTransformer
    
    # Create dummy data
    num_samples = 1000
    num_categorical = 20
    num_numerical = 26
    num_classes = 34
    batch_size = 128
    
    # Dummy categorical cardinalities
    categorical_cardinalities = [10, 5, 3, 100, 50] + [20] * 15
    
    # Generate dummy data with proper types
    # Categorical features: integers within cardinality bounds
    cat_data = []
    for cardinality in categorical_cardinalities:
        col = np.random.randint(0, cardinality, num_samples)
        cat_data.append(col)
    cat_data = np.column_stack(cat_data)  # [num_samples, num_categorical]
    
    # Numerical features: continuous values
    num_data = np.random.randn(num_samples, num_numerical)
    
    # Combine features
    X = np.column_stack([cat_data, num_data])  # [num_samples, num_categorical + num_numerical]
    y = np.random.randint(0, num_classes, num_samples)
    
    print(f"\n✅ Created dummy dataset:")
    print(f"   Samples: {num_samples}")
    print(f"   Features: categorical={num_categorical}, numerical={num_numerical}")
    print(f"   Classes: {num_classes}")
    
    # Create data loaders (simulate 3 clients)
    client_loaders = []
    for i in range(3):
        start_idx = i * (num_samples // 3)
        end_idx = (i + 1) * (num_samples // 3) if i < 2 else num_samples
        X_client = X[start_idx:end_idx]
        y_client = y[start_idx:end_idx]
        loader = create_data_loaders(X_client, y_client, batch_size=batch_size)
        client_loaders.append(loader)
        print(f"   Client {i}: {len(loader.dataset)} samples")
    
    # Create test loader
    test_loader = create_data_loaders(X[:200], y[:200], batch_size=batch_size, shuffle=False)
    
    # Create model
    print(f"\n🏗️  Creating TabTransformer model...")
    model = TabTransformer(
        categorical_cardinalities=categorical_cardinalities,
        num_numerical_features=num_numerical,
        num_classes=num_classes,
        embedding_dim=32,
        num_transformer_layers=2,
        num_attention_heads=4
    )
    print(f"   Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test FL training loop (just 2 rounds for testing)
    print(f"\n🚀 Running FL training (2 rounds for testing)...")
    history = federated_training_loop_pytorch(
        global_model=model,
        client_data_loaders=client_loaders,
        test_loader=test_loader,
        num_rounds=2,
        local_epochs=2,
        learning_rate=0.001,
        device='cpu',
        num_categorical=num_categorical,
        verbose=True
    )
    
    print(f"\n✅ FL training test completed!")
    print(f"   Final accuracy: {history['accuracy'][-1]*100:.2f}%")
    print(f"   History keys: {list(history.keys())}")
    
    print("\n" + "=" * 80)
    print("✅ All FL utility tests passed!")
    print("=" * 80)
