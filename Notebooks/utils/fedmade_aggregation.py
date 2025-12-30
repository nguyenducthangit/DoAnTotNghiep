"""
FedMade: Federated Model Aggregation with Dynamic Evaluation

This module implements an advanced federated aggregation strategy that uses
client performance metrics to compute dynamic contribution scores for weighted
model aggregation. This is more sophisticated than standard FedAvg.

Author: Nguyen Duc Thang
Project: IoT Network Attack Detection using Federated Learning

Note: The contribution scoring function is modular and can be updated later
      with different formulas or strategies.
"""

import torch
import numpy as np
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def default_contribution_score_function(
    client_metrics: List[Dict],
    accuracy_weight: float = 0.7,
    loss_weight: float = 0.3
) -> np.ndarray:
    """
    Default contribution scoring function.
    
    Computes scores based on weighted combination of accuracy and (inverted) loss.
    This function is modular and can be replaced with custom scoring logic.
    
    Args:
        client_metrics: List of dicts with {'accuracy', 'loss', 'num_samples'}
        accuracy_weight: Weight for accuracy component (default 0.7)
        loss_weight: Weight for loss component (default 0.3)
        
    Returns:
        Normalized contribution scores (sum to 1.0)
    """
    if len(client_metrics) == 0:
        return np.array([])
    
    # Extract metrics
    accuracies = np.array([m['accuracy'] for m in client_metrics])
    losses = np.array([m['loss'] for m in client_metrics])
    
    # Normalize losses (invert so lower loss = higher score)
    max_loss = losses.max()
    if max_loss > 0:
        normalized_losses = 1 - (losses / max_loss)
    else:
        normalized_losses = np.ones_like(losses)
    
    # Weighted combination
    scores = (accuracies * accuracy_weight) + (normalized_losses * loss_weight)
    
    # Normalize to sum to 1
    score_sum = scores.sum()
    if score_sum > 0:
        scores = scores / score_sum
    else:
        # Fallback: equal weights
        scores = np.ones(len(client_metrics)) / len(client_metrics)
    
    return scores


def compute_client_contribution_scores(
    client_metrics: List[Dict],
    scoring_function: Optional[Callable] = None,
    **scoring_kwargs
) -> np.ndarray:
    """
    Compute contribution scores for each client based on performance metrics.
    
    This is the main entry point for contribution scoring. It allows custom
    scoring functions to be plugged in.
    
    Args:
        client_metrics: List of client metrics dictionaries
        scoring_function: Custom scoring function (optional)
        **scoring_kwargs: Additional arguments for scoring function
        
    Returns:
        Array of contribution scores (normalized to sum to 1)
    """
    if scoring_function is None:
        scoring_function = default_contribution_score_function
    
    try:
        scores = scoring_function(client_metrics, **scoring_kwargs)
        
        # Validate scores
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)
        
        if len(scores) != len(client_metrics):
            raise ValueError(f"Score length {len(scores)} != metrics length {len(client_metrics)}")
        
        # Ensure non-negative and normalized
        scores = np.maximum(scores, 0)
        if scores.sum() > 0:
            scores = scores / scores.sum()
        else:
            scores = np.ones(len(client_metrics)) / len(client_metrics)
        
        return scores
        
    except Exception as e:
        logger.error(f"Error computing contribution scores: {e}. Using equal weights.")
        return np.ones(len(client_metrics)) / len(client_metrics)


def filter_low_contribution_clients(
    client_state_dicts: List[OrderedDict],
    client_metrics: List[Dict],
    contribution_scores: np.ndarray,
    threshold: float = 0.0
) -> Tuple[List[OrderedDict], List[Dict], np.ndarray, List[int]]:
    """
    Filter out clients with contribution scores below threshold.
    
    Args:
        client_state_dicts: List of model state_dicts
        client_metrics: List of client metrics
        contribution_scores: Contribution scores for each client
        threshold: Minimum score to include (0.0 means include all)
        
    Returns:
        Tuple of (filtered_state_dicts, filtered_metrics, renormalized_scores, valid_indices)
    """
    if threshold <= 0.0:
        # No filtering
        return client_state_dicts, client_metrics, contribution_scores, list(range(len(client_state_dicts)))
    
    # Find valid clients
    valid_indices = [i for i, score in enumerate(contribution_scores) if score >= threshold]
    
    if len(valid_indices) == 0:
        logger.warning(f"All clients filtered with threshold {threshold}. Including all clients.")
        return client_state_dicts, client_metrics, contribution_scores, list(range(len(client_state_dicts)))
    
    # Filter
    filtered_state_dicts = [client_state_dicts[i] for i in valid_indices]
    filtered_metrics = [client_metrics[i] for i in valid_indices]
    filtered_scores = contribution_scores[valid_indices]
    
    # Renormalize scores
    filtered_scores = filtered_scores / filtered_scores.sum()
    
    if len(valid_indices) < len(client_state_dicts):
        excluded = len(client_state_dicts) - len(valid_indices)
        logger.info(f"Filtered {excluded} clients with scores < {threshold}")
    
    return filtered_state_dicts, filtered_metrics, filtered_scores, valid_indices


def layer_wise_weighted_aggregation(
    client_state_dicts: List[OrderedDict],
    contribution_scores: np.ndarray
) -> OrderedDict:
    """
    Perform layer-wise weighted aggregation of client models.
    
    For each layer, computes weighted average using contribution scores.
    
    Args:
        client_state_dicts: List of state_dicts from clients
        contribution_scores: Normalized contribution scores
        
    Returns:
        Aggregated state_dict
    """
    if len(client_state_dicts) == 0:
        raise ValueError("No client state_dicts to aggregate")
    
    # Initialize aggregated state_dict
    aggregated_dict = OrderedDict()
    
    # Get layer names from first client
    layer_names = client_state_dicts[0].keys()
    
    # Aggregate each layer
    for layer_name in layer_names:
        # Initialize weighted sum
        weighted_sum = None
        
        for i, state_dict in enumerate(client_state_dicts):
            if layer_name not in state_dict:
                raise ValueError(f"Layer {layer_name} missing in client {i} state_dict")
            
            layer_tensor = state_dict[layer_name]
            
            if weighted_sum is None:
                weighted_sum = contribution_scores[i] * layer_tensor
            else:
                weighted_sum += contribution_scores[i] * layer_tensor
        
        aggregated_dict[layer_name] = weighted_sum
    
    return aggregated_dict


def fedmade_aggregate(
    client_state_dicts: List[OrderedDict],
    client_metrics: List[Dict],
    global_model: Optional[torch.nn.Module] = None,
    contribution_threshold: float = 0.0,
    accuracy_weight: float = 0.7,
    loss_weight: float = 0.3,
    scoring_function: Optional[Callable] = None,
    verbose: bool = False
) -> OrderedDict:
    """
    FedMade aggregation: Dynamic weighted aggregation based on client performance.
    
    This is the main FedMade aggregation function that orchestrates:
    1. Computing contribution scores
    2. Filtering low-quality clients (optional)
    3. Layer-wise weighted aggregation
    
    Args:
        client_state_dicts: List of state_dicts from clients
        client_metrics: List of metrics dicts with {'accuracy', 'loss', 'num_samples'}
        global_model: Current global model (optional, for reference)
        contribution_threshold: Minimum score to include client (0.0 = no filtering)
        accuracy_weight: Weight for accuracy in scoring (default 0.7)
        loss_weight: Weight for loss in scoring (default 0.3)
        scoring_function: Custom scoring function (optional)
        verbose: Log detailed information
        
    Returns:
        Aggregated state_dict
    """
    if len(client_state_dicts) == 0:
        raise ValueError("No client models to aggregate")
    
    if len(client_state_dicts) != len(client_metrics):
        raise ValueError(f"Mismatch: {len(client_state_dicts)} state_dicts but {len(client_metrics)} metrics")
    
    # Step 1: Compute contribution scores
    scores = compute_client_contribution_scores(
        client_metrics,
        scoring_function=scoring_function,
        accuracy_weight=accuracy_weight,
        loss_weight=loss_weight
    )
    
    if verbose:
        logger.info("Client Contribution Scores:")
        for i, (score, metrics) in enumerate(zip(scores, client_metrics)):
            logger.info(f"  Client {i}: score={score:.4f}, acc={metrics['accuracy']:.4f}, loss={metrics['loss']:.4f}")
    
    # Step 2: Filter low-quality clients
    filtered_dicts, filtered_metrics, filtered_scores, valid_indices = filter_low_contribution_clients(
        client_state_dicts,
        client_metrics,
        scores,
        threshold=contribution_threshold
    )
    
    if verbose and len(valid_indices) < len(client_state_dicts):
        excluded_clients = set(range(len(client_state_dicts))) - set(valid_indices)
        logger.info(f"Excluded clients: {excluded_clients}")
    
    # Step 3: Layer-wise weighted aggregation
    aggregated_dict = layer_wise_weighted_aggregation(
        filtered_dicts,
        filtered_scores
    )
    
    if verbose:
        logger.info(f"FedMade aggregation complete: {len(filtered_dicts)}/{len(client_state_dicts)} clients included")
    
    return aggregated_dict


def fedmade_aggregate_with_fallback(
    client_state_dicts: List[OrderedDict],
    client_metrics: List[Dict],
    client_weights: Optional[List[int]] = None,
    **fedmade_kwargs
) -> OrderedDict:
    """
    FedMade aggregation with automatic fallback to FedAvg on error.
    
    This is a safe wrapper around fedmade_aggregate that falls back to
    standard FedAvg if FedMade encounters errors.
    
    Args:
        client_state_dicts: List of state_dicts from clients
        client_metrics: List of metrics dicts
        client_weights: Sample counts for FedAvg fallback (optional)
        **fedmade_kwargs: Arguments for fedmade_aggregate
        
    Returns:
        Aggregated state_dict
    """
    try:
        return fedmade_aggregate(client_state_dicts, client_metrics, **fedmade_kwargs)
        
    except Exception as e:
        logger.error(f"FedMade aggregation failed: {e}. Falling back to FedAvg.")
        
        # Fallback to FedAvg
        if client_weights is None:
            # Use equal weights
            client_weights = [1] * len(client_state_dicts)
        
        return fedavg_fallback(client_state_dicts, client_weights)


def fedavg_fallback(
    client_state_dicts: List[OrderedDict],
    client_weights: List[int]
) -> OrderedDict:
    """
    Standard FedAvg aggregation (fallback).
    
    Args:
        client_state_dicts: List of state_dicts
        client_weights: Number of samples per client
        
    Returns:
        Aggregated state_dict
    """
    if len(client_state_dicts) == 0:
        raise ValueError("No client models to aggregate")
    
    total_weight = sum(client_weights)
    aggregated_dict = OrderedDict()
    
    layer_names = client_state_dicts[0].keys()
    
    for layer_name in layer_names:
        weighted_sum = None
        
        for i, state_dict in enumerate(client_state_dicts):
            weight = client_weights[i] / total_weight
            layer_tensor = state_dict[layer_name]
            
            if weighted_sum is None:
                weighted_sum = weight * layer_tensor
            else:
                weighted_sum += weight * layer_tensor
        
        aggregated_dict[layer_name] = weighted_sum
    
    return aggregated_dict


def log_contribution_scores(
    contribution_scores: np.ndarray,
    client_metrics: List[Dict],
    round_num: int,
    save_path: Optional[str] = None
) -> Dict:
    """
    Log contribution scores to console and optionally save to file.
    
    Args:
        contribution_scores: Contribution scores
        client_metrics: Client metrics
        round_num: Current training round
        save_path: Path to save logs (optional)
        
    Returns:
        Log dictionary
    """
    log_data = {
        'round': round_num,
        'clients': []
    }
    
    for i, (score, metrics) in enumerate(zip(contribution_scores, client_metrics)):
        client_log = {
            'client_id': i,
            'contribution_score': float(score),
            'accuracy': float(metrics['accuracy']),
            'loss': float(metrics['loss']),
            'num_samples': int(metrics.get('num_samples', 0))
        }
        log_data['clients'].append(client_log)
    
    if save_path:
        import json
        try:
            # Load existing logs
            with open(save_path, 'r') as f:
                logs = json.load(f)
        except FileNotFoundError:
            logs = []
        
        logs.append(log_data)
        
        with open(save_path, 'w') as f:
            json.dump(logs, f, indent=2)
    
    return log_data


if __name__ == '__main__':
    # Test FedMade aggregation
    logger.info("Testing FedMade aggregation module")
    
    # Create mock client state_dicts
    num_clients = 5
    client_state_dicts = []
    
    for i in range(num_clients):
        state_dict = OrderedDict()
        state_dict['layer1.weight'] = torch.randn(10, 5)
        state_dict['layer1.bias'] = torch.randn(10)
        state_dict['layer2.weight'] = torch.randn(3, 10)
        state_dict['layer2.bias'] = torch.randn(3)
        client_state_dicts.append(state_dict)
    
    # Create mock client metrics (varying quality)
    client_metrics = [
        {'accuracy': 0.92, 'loss': 0.15, 'num_samples': 1000},  # Good client
        {'accuracy': 0.75, 'loss': 0.45, 'num_samples': 800},   # Poor client
        {'accuracy': 0.88, 'loss': 0.22, 'num_samples': 1200},  # Good client
        {'accuracy': 0.65, 'loss': 0.60, 'num_samples': 600},   # Poor client
        {'accuracy': 0.90, 'loss': 0.18, 'num_samples': 1100},  # Good client
    ]
    
    print(f"\n{'='*60}")
    print(f"Test 1: FedMade without filtering")
    print(f"{'='*60}")
    
    aggregated = fedmade_aggregate(
        client_state_dicts,
        client_metrics,
        contribution_threshold=0.0,
        verbose=True
    )
    
    print(f"\nAggregated model layers: {list(aggregated.keys())}")
    print(f"Layer1 weight shape: {aggregated['layer1.weight'].shape}")
    
    print(f"\n{'='*60}")
    print(f"Test 2: FedMade with filtering (threshold=0.25)")
    print(f"{'='*60}")
    
    aggregated_filtered = fedmade_aggregate(
        client_state_dicts,
        client_metrics,
        contribution_threshold=0.25,
        verbose=True
    )
    
    print(f"\n{'='*60}")
    print(f"Test 3: FedAvg fallback")
    print(f"{'='*60}")
    
    client_weights = [m['num_samples'] for m in client_metrics]
    aggregated_fedavg = fedavg_fallback(client_state_dicts, client_weights)
    
    print(f"FedAvg aggregation complete")
    
    print(f"\n✅ FedMade module test passed!")
