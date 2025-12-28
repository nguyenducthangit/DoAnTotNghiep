"""
Federated Learning Utilities for IoT Attack Detection

This module implements the Federated Learning logic including:
- FederatedClient: Local training on client data
- FederatedServer: Global model aggregation using FedAvg
- Training loop orchestration

Author: Nguyen Duc Thang
Project: IoT Network Attack Detection using Federated Learning
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Tuple
import copy


class FederatedClient:
    """
    Federated Learning Client that trains model on local data.
    """
    
    def __init__(self, client_id: int, X_train: np.ndarray, y_train: np.ndarray):
        """
        Initialize a federated client.
        
        Args:
            client_id: Unique identifier for this client
            X_train: Training features for this client
            y_train: Training labels for this client
        """
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.model = None
        
        print(f"   Client {client_id} initialized with {len(X_train):,} samples")
    
    def set_model(self, model: keras.Model):
        """
        Set the model for this client (receives from server).
        
        Args:
            model: Keras model with global weights
        """
        self.model = model
    
    def local_train(self, epochs: int = 5, batch_size: int = 256, 
                   verbose: int = 0) -> Tuple[List[np.ndarray], Dict]:
        """
        Train the model on local data.
        
        Args:
            epochs: Number of local training epochs
            batch_size: Batch size for training
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
            
        Returns:
            Tuple of (updated weights, training history)
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        # Train on local data
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=0.1  # Use 10% for local validation
        )
        
        # Get updated weights
        updated_weights = self.model.get_weights()
        
        # Extract history
        history_dict = {
            'loss': history.history['loss'],
            'accuracy': history.history['accuracy'],
            'val_loss': history.history.get('val_loss', []),
            'val_accuracy': history.history.get('val_accuracy', [])
        }
        
        return updated_weights, history_dict
    
    def get_weights(self) -> List[np.ndarray]:
        """
        Get current model weights.
        
        Returns:
            List of weight arrays
        """
        if self.model is None:
            raise ValueError("Model not set.")
        return self.model.get_weights()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Tuple of (loss, accuracy)
        """
        if self.model is None:
            raise ValueError("Model not set.")
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy


class FederatedServer:
    """
    Federated Learning Server that aggregates client updates using FedAvg.
    """
    
    def __init__(self, model: keras.Model):
        """
        Initialize the federated server with a global model.
        
        Args:
            model: Initial global model
        """
        self.global_model = model
        self.training_history = {
            'round': [],
            'loss': [],
            'accuracy': [],
            'client_losses': [],
            'client_accuracies': []
        }
        
        print(f"🖥️  Federated Server initialized")
    
    def get_global_model(self) -> keras.Model:
        """
        Get a copy of the current global model with a FRESH optimizer.
        
        This method fixes the KeyError issue by creating a completely new
        optimizer instance for each client model copy.
        
        Returns:
            Copy of global model with fresh optimizer
        """
        # Clone model architecture
        model_copy = keras.models.clone_model(self.global_model)
        
        # Copy weights from global model
        model_copy.set_weights(self.global_model.get_weights())
        
        # Create FRESH optimizer instance (critical fix!)
        # This ensures the optimizer isn't tied to old variable names
        optimizer_config = self.global_model.optimizer.get_config()
        optimizer_class = type(self.global_model.optimizer)
        fresh_optimizer = optimizer_class.from_config(optimizer_config)
        
        # Compile with fresh optimizer
        model_copy.compile(
            optimizer=fresh_optimizer,
            loss=self.global_model.loss,
            metrics=['accuracy']  # Use simple string to avoid compiled_metrics issues
        )
        
        return model_copy
    
    def aggregate_weights(self, client_weights: List[List[np.ndarray]]) -> List[np.ndarray]:
        """
        Aggregate client weights using Federated Averaging (FedAvg).
        
        FedAvg: Simply average all client weights element-wise.
        
        Args:
            client_weights: List of weight lists from all clients
            
        Returns:
            Aggregated weights
        """
        # Initialize aggregated weights with zeros
        num_layers = len(client_weights[0])
        aggregated_weights = []
        
        for layer_idx in range(num_layers):
            # Stack weights from all clients for this layer
            layer_weights = np.array([client[layer_idx] for client in client_weights])
            
            # Average across clients
            avg_weights = np.mean(layer_weights, axis=0)
            aggregated_weights.append(avg_weights)
        
        return aggregated_weights
    
    def update_global_model(self, aggregated_weights: List[np.ndarray]):
        """
        Update the global model with aggregated weights.
        
        Args:
            aggregated_weights: Aggregated weights from FedAvg
        """
        self.global_model.set_weights(aggregated_weights)
    
    def evaluate_global_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate the global model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Tuple of (loss, accuracy)
        """
        loss, accuracy = self.global_model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy
    
    def record_round(self, round_num: int, loss: float, accuracy: float,
                    client_losses: List[float] = None, 
                    client_accuracies: List[float] = None):
        """
        Record training metrics for this round.
        
        Args:
            round_num: Current round number
            loss: Global model loss on test set
            accuracy: Global model accuracy on test set
            client_losses: List of client losses (optional)
            client_accuracies: List of client accuracies (optional)
        """
        self.training_history['round'].append(round_num)
        self.training_history['loss'].append(loss)
        self.training_history['accuracy'].append(accuracy)
        
        if client_losses:
            self.training_history['client_losses'].append(client_losses)
        if client_accuracies:
            self.training_history['client_accuracies'].append(client_accuracies)
    
    def get_history(self) -> Dict:
        """
        Get training history.
        
        Returns:
            Dictionary with training history
        """
        return self.training_history
    
    def save_model(self, save_path: str):
        """
        Save the global model.
        
        Args:
            save_path: Path to save the model
        """
        self.global_model.save(save_path)
        print(f"   💾 Global model saved to: {save_path}")


def federated_training_loop(server: FederatedServer,
                            clients: List[FederatedClient],
                            X_test: np.ndarray,
                            y_test: np.ndarray,
                            num_rounds: int = 30,
                            local_epochs: int = 5,
                            batch_size: int = 256,
                            verbose: int = 1) -> Dict:
    """
    Main federated learning training loop.
    
    Args:
        server: FederatedServer instance
        clients: List of FederatedClient instances
        X_test: Test features for evaluation
        y_test: Test labels for evaluation
        num_rounds: Number of FL communication rounds
        local_epochs: Number of local training epochs per round
        batch_size: Batch size for local training
        verbose: Verbosity level
        
    Returns:
        Training history dictionary
    """
    print("\n" + "=" * 80)
    print("FEDERATED LEARNING TRAINING")
    print("=" * 80)
    print(f"Number of clients: {len(clients)}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Local epochs per round: {local_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Test set size: {len(X_test):,}")
    print("=" * 80 + "\n")
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*80}")
        print(f"ROUND {round_num}/{num_rounds}")
        print(f"{'='*80}")
        
        # Step 1: Broadcast global model to all clients
        if verbose >= 1:
            print(f"📡 Broadcasting global model to {len(clients)} clients...")
        
        for client in clients:
            client.set_model(server.get_global_model())
        
        # Step 2: Local training on each client
        client_weights = []
        client_losses = []
        client_accuracies = []
        
        for i, client in enumerate(clients):
            if verbose >= 1:
                print(f"\n   Client {client.client_id} training...", end=" ")
            
            # Train locally
            weights, history = client.local_train(
                epochs=local_epochs,
                batch_size=batch_size,
                verbose=0  # Silent for cleaner output
            )
            
            client_weights.append(weights)
            
            # Record client metrics (last epoch)
            final_loss = history['loss'][-1]
            final_acc = history['accuracy'][-1]
            client_losses.append(final_loss)
            client_accuracies.append(final_acc)
            
            if verbose >= 1:
                print(f"✓ Loss: {final_loss:.4f}, Acc: {final_acc:.4f}")
        
        # Step 3: Aggregate weights using FedAvg
        if verbose >= 1:
            print(f"\n🔄 Aggregating weights from {len(clients)} clients...")
        
        aggregated_weights = server.aggregate_weights(client_weights)
        
        # Step 4: Update global model
        server.update_global_model(aggregated_weights)
        
        if verbose >= 1:
            print(f"   ✓ Global model updated")
        
        # Step 5: Evaluate global model on test set
        if verbose >= 1:
            print(f"\n📊 Evaluating global model on test set...")
        
        test_loss, test_accuracy = server.evaluate_global_model(X_test, y_test)
        
        # Record metrics
        server.record_round(
            round_num=round_num,
            loss=test_loss,
            accuracy=test_accuracy,
            client_losses=client_losses,
            client_accuracies=client_accuracies
        )
        
        # Print round summary
        print(f"\n{'─'*80}")
        print(f"ROUND {round_num} SUMMARY:")
        print(f"   Global Test Loss: {test_loss:.4f}")
        print(f"   Global Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"   Avg Client Loss: {np.mean(client_losses):.4f}")
        print(f"   Avg Client Accuracy: {np.mean(client_accuracies):.4f}")
        print(f"{'─'*80}")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED!")
    print(f"{'='*80}")
    print(f"Final Test Accuracy: {server.training_history['accuracy'][-1]*100:.2f}%")
    print(f"{'='*80}\n")
    
    return server.get_history()


if __name__ == '__main__':
    # Test the classes
    print("=" * 80)
    print("Testing Federated Learning Utilities")
    print("=" * 80)
    
    print("\n✅ FederatedClient and FederatedServer classes defined successfully!")
    print("   Use these classes in 2_Federated_Training.ipynb")
