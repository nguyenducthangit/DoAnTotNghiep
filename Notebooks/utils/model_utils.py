"""
Model Utilities for Federated Learning IoT Attack Detection

This module provides functions for creating, compiling, and manipulating
the Deep Neural Network model for attack classification.

Author: Nguyen Duc Thang
Project: IoT Network Attack Detection using Federated Learning
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from typing import List, Tuple


def create_dnn_model(input_dim: int = 46,
                     hidden_layers: List[int] = [128, 64, 32],
                     num_classes: int = 34,
                     dropout_rate: float = 0.3,
                     activation: str = 'relu',
                     output_activation: str = 'softmax') -> keras.Model:
    """
    Create a Deep Neural Network model for multi-class attack classification.
    
    Architecture:
        Input (input_dim) 
        → Dense(hidden_layers[0], activation) + Dropout(dropout_rate)
        → Dense(hidden_layers[1], activation) + Dropout(dropout_rate)
        → Dense(hidden_layers[2], activation) + Dropout(dropout_rate)
        → Dense(num_classes, output_activation)
    
    Args:
        input_dim: Number of input features
        hidden_layers: List of hidden layer sizes
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        activation: Activation function for hidden layers
        output_activation: Activation function for output layer
        
    Returns:
        Compiled Keras model
    """
    print(f"\n🏗️  Creating DNN model...")
    print(f"   Input dimension: {input_dim}")
    print(f"   Hidden layers: {hidden_layers}")
    print(f"   Output classes: {num_classes}")
    print(f"   Dropout rate: {dropout_rate}")
    
    model = Sequential(name='IoT_Attack_Detection_DNN')
    
    # Input layer + First hidden layer
    model.add(Dense(hidden_layers[0], activation=activation, 
                   input_dim=input_dim, name='dense_1'))
    model.add(Dropout(dropout_rate, name='dropout_1'))
    
    # Additional hidden layers
    for i, units in enumerate(hidden_layers[1:], start=2):
        model.add(Dense(units, activation=activation, name=f'dense_{i}'))
        model.add(Dropout(dropout_rate, name=f'dropout_{i}'))
    
    # Output layer
    model.add(Dense(num_classes, activation=output_activation, name='output'))
    
    print(f"   ✓ Model created with {len(hidden_layers)} hidden layers")
    
    return model


def compile_model(model: keras.Model,
                 learning_rate: float = 0.001,
                 loss: str = 'sparse_categorical_crossentropy',
                 metrics: List[str] = ['accuracy']) -> keras.Model:
    """
    Compile the model with optimizer, loss function, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
        loss: Loss function
        metrics: List of metrics to track
        
    Returns:
        Compiled model
    """
    print(f"\n⚙️  Compiling model...")
    print(f"   Optimizer: Adam (lr={learning_rate})")
    print(f"   Loss: {loss}")
    print(f"   Metrics: {metrics}")
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    print(f"   ✓ Model compiled successfully")
    
    return model


def get_model_weights(model: keras.Model) -> List[np.ndarray]:
    """
    Extract weights from a Keras model.
    
    Args:
        model: Keras model
        
    Returns:
        List of numpy arrays representing layer weights
    """
    return model.get_weights()


def set_model_weights(model: keras.Model, weights: List[np.ndarray]) -> keras.Model:
    """
    Set weights for a Keras model.
    
    Args:
        model: Keras model
        weights: List of numpy arrays representing layer weights
        
    Returns:
        Model with updated weights
    """
    model.set_weights(weights)
    return model


def print_model_summary(model: keras.Model):
    """
    Print detailed model summary.
    
    Args:
        model: Keras model
    """
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 80)
    model.summary()
    print("=" * 80)
    
    # Calculate total parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    # Estimate model size
    model_size_mb = (total_params * 4) / (1024 * 1024)  # Assuming float32 (4 bytes)
    print(f"Estimated model size: {model_size_mb:.2f} MB")
    print("=" * 80 + "\n")


def save_model(model: keras.Model, save_path: str):
    """
    Save model to file.
    
    Args:
        model: Keras model to save
        save_path: Path to save the model (.h5 file)
    """
    print(f"\n💾 Saving model to: {save_path}")
    model.save(save_path)
    print(f"   ✓ Model saved successfully")


def load_model(model_path: str) -> keras.Model:
    """
    Load model from file.
    
    Args:
        model_path: Path to the saved model (.h5 file)
        
    Returns:
        Loaded Keras model
    """
    print(f"\n📂 Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print(f"   ✓ Model loaded successfully")
    return model


def create_and_compile_model(config: dict) -> keras.Model:
    """
    Create and compile model from configuration dictionary.
    
    Args:
        config: Configuration dictionary with model and optimizer settings
        
    Returns:
        Compiled Keras model
    """
    # Extract model config
    model_config = config.get('model', {})
    input_dim = model_config.get('input_dim', 46)
    hidden_layers = model_config.get('hidden_layers', [128, 64, 32])
    num_classes = model_config.get('num_classes', 34)
    dropout_rate = model_config.get('dropout_rate', 0.3)
    activation = model_config.get('activation', 'relu')
    output_activation = model_config.get('output_activation', 'softmax')
    
    # Extract optimizer config
    optimizer_config = config.get('optimizer', {})
    learning_rate = optimizer_config.get('learning_rate', 0.001)
    
    # Extract training config
    loss = config.get('loss_function', 'sparse_categorical_crossentropy')
    metrics = config.get('metrics', ['accuracy'])
    
    # Create model
    model = create_dnn_model(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        activation=activation,
        output_activation=output_activation
    )
    
    # Compile model
    model = compile_model(
        model=model,
        learning_rate=learning_rate,
        loss=loss,
        metrics=metrics
    )
    
    return model


if __name__ == '__main__':
    # Test the functions
    print("=" * 80)
    print("Testing Model Utilities")
    print("=" * 80)
    
    # Create a test model
    test_model = create_dnn_model(
        input_dim=46,
        hidden_layers=[128, 64, 32],
        num_classes=34,
        dropout_rate=0.3
    )
    
    # Compile the model
    test_model = compile_model(test_model, learning_rate=0.001)
    
    # Print summary
    print_model_summary(test_model)
    
    # Test weight extraction
    weights = get_model_weights(test_model)
    print(f"✓ Extracted {len(weights)} weight arrays")
    
    # Test weight setting
    test_model = set_model_weights(test_model, weights)
    print(f"✓ Set weights successfully")
    
    print("\n✅ All model utility functions tested successfully!")
    print("   Use these functions in 2_Federated_Training.ipynb")
