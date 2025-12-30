"""
TabTransformer Model for Federated Learning IoT Attack Detection

This module provides PyTorch implementation of TabTransformer architecture
specifically designed for tabular network attack classification.

Author: Nguyen Duc Thang
Project: IoT Network Attack Detection using Federated Learning
Framework: PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class FeatureEmbedding(nn.Module):
    """
    Embedding layer that handles both categorical and numerical features.
    
    Categorical features are embedded using learned embedding tables.
    Numerical features are projected through a linear layer.
    """
    
    def __init__(self,
                 categorical_cardinalities: List[int],
                 num_numerical_features: int,
                 embedding_dim: int = 32):
        """
        Args:
            categorical_cardinalities: List of unique values count for each categorical feature
            num_numerical_features: Number of numerical features
            embedding_dim: Dimension of embeddings (same for categorical and numerical)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_categorical = len(categorical_cardinalities)
        self.num_numerical = num_numerical_features
        
        # Categorical embeddings - one embedding table per categorical feature
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=cardinality, embedding_dim=embedding_dim)
            for cardinality in categorical_cardinalities
        ])
        
        # Numerical projection - shared linear layer for all numerical features
        if num_numerical_features > 0:
            self.numerical_projection = nn.Linear(1, embedding_dim)
            self.numerical_norm = nn.LayerNorm(embedding_dim)
        else:
            self.numerical_projection = None
    
    def forward(self, categorical_features: Optional[torch.Tensor] = None,
                numerical_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through embedding layer.
        
        Args:
            categorical_features: [batch_size, num_categorical] - integer encoded
            numerical_features: [batch_size, num_numerical] - float values
            
        Returns:
            embeddings: [batch_size, total_features, embedding_dim]
        """
        embeddings_list = []
        
        # Embed categorical features
        if (categorical_features is not None and 
            self.num_categorical > 0 and 
            categorical_features.size(1) > 0):
            # Process each categorical feature separately
            # Only process as many features as are available
            num_available = min(self.num_categorical, categorical_features.size(1))
            for i in range(num_available):
                # Get column i from categorical features
                cat_col = categorical_features[:, i].long()
                embedded = self.categorical_embeddings[i](cat_col)  # [batch_size, embedding_dim]
                embeddings_list.append(embedded)
        
        # Project numerical features
        if (numerical_features is not None and 
            self.num_numerical > 0 and 
            numerical_features.size(1) > 0):
            # Process each numerical feature separately
            # Only process as many features as are available
            num_available = min(self.num_numerical, numerical_features.size(1))
            for i in range(num_available):
                num_col = numerical_features[:, i:i+1]  # [batch_size, 1]
                projected = self.numerical_projection(num_col)  # [batch_size, embedding_dim]
                projected = self.numerical_norm(projected)
                embeddings_list.append(projected)
        
        # Stack all embeddings: [batch_size, total_features, embedding_dim]
        embeddings = torch.stack(embeddings_list, dim=1)
        
        return embeddings


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer Encoder block with multi-head attention and feed-forward network.
    Includes residual connections and layer normalization.
    """
    
    def __init__(self,
                 embed_dim: int = 32,
                 num_heads: int = 4,
                 ff_hidden_dim: int = 128,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_hidden_dim: Hidden dimension in feed-forward network
            dropout: Dropout rate
            attention_dropout: Dropout rate for attention weights
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True  # Input shape: [batch, seq, embed]
        )
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer encoder block.
        
        Args:
            x: [batch_size, seq_len, embed_dim]
            
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        # Multi-head attention with residual connection
        attended, _ = self.attention(x, x, x)  # Self-attention
        x = self.norm1(x + self.dropout(attended))
        
        # Feed-forward network with residual connection
        ff_output = self.ff_network(x)
        x = self.norm2(x + ff_output)
        
        return x


class TabTransformer(nn.Module):
    """
    TabTransformer model for tabular data classification.
    
    Architecture:
    1. Feature Embedding: Separate pathways for categorical and numerical features
    2. Transformer Encoder: Multi-layer self-attention for feature interactions
    3. Classification Head: MLP for final prediction
    """
    
    def __init__(self,
                 categorical_cardinalities: List[int],
                 num_numerical_features: int,
                 num_classes: int = 34,
                 embedding_dim: int = 32,
                 num_transformer_layers: int = 2,
                 num_attention_heads: int = 4,
                 ff_hidden_dim: int = 128,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 use_cls_token: bool = True):
        """
        Args:
            categorical_cardinalities: List of unique values for each categorical feature
            num_numerical_features: Number of numerical features
            num_classes: Number of output classes
            embedding_dim: Dimension of embeddings
            num_transformer_layers: Number of transformer encoder layers
            num_attention_heads: Number of attention heads per layer
            ff_hidden_dim: Hidden dimension in feed-forward network
            dropout: Dropout rate
            attention_dropout: Dropout for attention weights
            use_cls_token: Whether to use CLS token for classification
        """
        super().__init__()
        
        self.num_categorical = len(categorical_cardinalities)
        self.num_numerical = num_numerical_features
        self.total_features = self.num_categorical + self.num_numerical
        self.embedding_dim = embedding_dim
        self.use_cls_token = use_cls_token
        
        # Feature embedding layer
        self.feature_embedding = FeatureEmbedding(
            categorical_cardinalities=categorical_cardinalities,
            num_numerical_features=num_numerical_features,
            embedding_dim=embedding_dim
        )
        
        # CLS token (optional - used for classification)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # Transformer encoder stack
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embedding_dim,
                num_heads=num_attention_heads,
                ff_hidden_dim=ff_hidden_dim,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
            for _ in range(num_transformer_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, 
                categorical_features: Optional[torch.Tensor] = None,
                numerical_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through TabTransformer.
        
        Args:
            categorical_features: [batch_size, num_categorical] - integer encoded
            numerical_features: [batch_size, num_numerical] - float values
            
        Returns:
            logits: [batch_size, num_classes] - class logits (before softmax)
        """
        batch_size = categorical_features.size(0) if categorical_features is not None else numerical_features.size(0)
        
        # 1. Embed features
        x = self.feature_embedding(categorical_features, numerical_features)
        # x shape: [batch_size, total_features, embedding_dim]
        
        # 2. Add CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            # x shape: [batch_size, total_features + 1, embedding_dim]
        
        # 3. Pass through transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        
        # 4. Extract representation for classification
        if self.use_cls_token:
            # Use CLS token representation
            representation = x[:, 0, :]  # [batch_size, embedding_dim]
        else:
            # Use mean pooling over all features
            representation = x.mean(dim=1)  # [batch_size, embedding_dim]
        
        # 5. Classification head
        logits = self.classifier(representation)  # [batch_size, num_classes]
        
        return logits
    
    def get_attention_weights(self) -> List[torch.Tensor]:
        """
        Extract attention weights from all transformer layers.
        Useful for interpretability.
        
        Returns:
            List of attention weight tensors, one per layer
        """
        attention_weights = []
        for layer in self.transformer_layers:
            # Note: This requires storing attention weights during forward pass
            # For now, return empty list (can be enhanced later)
            pass
        return attention_weights


def create_tabtransformer_from_config(config: dict) -> TabTransformer:
    """
    Create TabTransformer model from configuration dictionary.
    
    Args:
        config: Configuration dict with model hyperparameters
        
    Returns:
        TabTransformer model instance
    """
    # Extract TabTransformer config
    tt_config = config.get('tabtransformer', {})
    model_config = config.get('model', {})
    
    # Get feature configuration
    feature_config = config.get('features', {})
    categorical_cardinalities = feature_config.get('categorical_cardinalities', [])
    num_categorical = len(categorical_cardinalities)
    
    # If no explicit split, assume based on typical network data (20 categorical, 26 numerical)
    if num_categorical == 0:
        print("⚠️  No categorical features specified. Using default split (20 cat, 26 num)")
        categorical_cardinalities = [50] * 20  # Default: 20 categorical with avg 50 unique values
    
    total_features = model_config.get('input_dim', 46)
    num_numerical = total_features - num_categorical
    
    # Create model
    model = TabTransformer(
        categorical_cardinalities=categorical_cardinalities,
        num_numerical_features=num_numerical,
        num_classes=model_config.get('num_classes', 34),
        embedding_dim=tt_config.get('embedding_dim', 32),
        num_transformer_layers=tt_config.get('num_transformer_layers', 2),
        num_attention_heads=tt_config.get('num_attention_heads', 4),
        ff_hidden_dim=tt_config.get('ff_hidden_dim', 128),
        dropout=tt_config.get('dropout_rate', 0.1),
        attention_dropout=tt_config.get('attention_dropout', 0.1),
        use_cls_token=tt_config.get('use_cls_token', True)
    )
    
    print(f"\n🏗️  Created TabTransformer model:")
    print(f"   Categorical features: {num_categorical}")
    print(f"   Numerical features: {num_numerical}")
    print(f"   Total features: {total_features}")
    print(f"   Embedding dimension: {tt_config.get('embedding_dim', 32)}")
    print(f"   Transformer layers: {tt_config.get('num_transformer_layers', 2)}")
    print(f"   Attention heads: {tt_config.get('num_attention_heads', 4)}")
    print(f"   Output classes: {model_config.get('num_classes', 34)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024:.2f} KB (FP32)")
    
    return model


def save_model_pytorch(model: nn.Module, save_path: str):
    """
    Save PyTorch model to file.
    
    Args:
        model: PyTorch model
        save_path: Path to save model (.pth or .pt file)
    """
    print(f"\n💾 Saving PyTorch model to: {save_path}")
    torch.save(model.state_dict(), save_path)
    print(f"   ✓ Model saved successfully")


def load_model_pytorch(model: nn.Module, model_path: str, device: str = 'cpu') -> nn.Module:
    """
    Load PyTorch model from file.
    
    Args:
        model: Model instance with same architecture
        model_path: Path to saved model (.pth or .pt file)
        device: Device to load model to ('cpu' or 'cuda')
        
    Returns:
        Model with loaded weights
    """
    print(f"\n📂 Loading PyTorch model from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"   ✓ Model loaded successfully on {device}")
    return model


if __name__ == '__main__':
    # Test the TabTransformer implementation
    print("=" * 80)
    print("Testing TabTransformer Implementation")
    print("=" * 80)
    
    # Test configuration
    batch_size = 16
    num_categorical = 20
    num_numerical = 26
    num_classes = 34
    
    # Create dummy categorical cardinalities (typical for network data)
    categorical_cardinalities = [10, 5, 3, 100, 50] + [20] * 15  # 20 categorical features
    
    # Create model
    model = TabTransformer(
        categorical_cardinalities=categorical_cardinalities,
        num_numerical_features=num_numerical,
        num_classes=num_classes,
        embedding_dim=32,
        num_transformer_layers=2,
        num_attention_heads=4,
        ff_hidden_dim=128,
        dropout=0.1
    )
    
    print(f"\n✅ Model created successfully")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input - respect individual cardinalities
    cat_features_list = []
    for cardinality in categorical_cardinalities:
        # Generate features in range [0, cardinality-1]
        cat_col = torch.randint(0, cardinality, (batch_size,))
        cat_features_list.append(cat_col)
    cat_features = torch.stack(cat_features_list, dim=1)
    num_features = torch.randn(batch_size, num_numerical)
    
    print(f"\n🔍 Testing forward pass...")
    print(f"   Input shapes:")
    print(f"   - Categorical: {cat_features.shape}")
    print(f"   - Numerical: {num_features.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(cat_features, num_features)
    
    print(f"\n✅ Forward pass successful!")
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Expected shape: [{batch_size}, {num_classes}]")
    
    # Test softmax
    probabilities = F.softmax(logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    
    print(f"\n📊 Predictions:")
    print(f"   Probabilities shape: {probabilities.shape}")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Sample predictions (first 5): {predictions[:5].tolist()}")
    print(f"   Probability sums (should be ~1.0): {probabilities.sum(dim=1)[:5].tolist()}")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed! TabTransformer is ready for FL training.")
    print("=" * 80)
