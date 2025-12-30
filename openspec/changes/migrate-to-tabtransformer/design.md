# Technical Design: TabTransformer for Federated Learning

## Architecture Overview

### TabTransformer Model Structure

The TabTransformer architecture consists of three main components:

1. **Feature Embedding Layer**: Processes categorical and numerical features differently
2. **Transformer Encoder**: Multi-layer attention mechanism for feature interaction learning  
3. **Classification Head**: MLP layers for final prediction

```
┌─────────────────────────────────────────────────────────┐
│                    Input Features (46)                   │
│           Protocol, Port, Flags, Packet Stats, etc.      │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴────────────┐
         │                        │
    Categorical              Numerical
    Features (20)           Features (26)
         │                        │
         │                        │
    ┌────▼────┐              ┌────▼────┐
    │Embedding│              │ Linear  │
    │  Layer  │              │ Projection│
    │ (32-dim)│              │ (32-dim)│
    └────┬────┘              └────┬────┘
         │                        │
         └────────┬───────────────┘
                  │
         [Batch, 46, 32] Embeddings
                  │
         ┌────────▼────────┐
         │  Add [CLS]      │
         │  Token          │
         └────────┬────────┘
                  │
         [Batch, 47, 32]
                  │
    ┌─────────────▼──────────────┐
    │  Transformer Encoder       │
    │  ┌──────────────────────┐  │
    │  │ Multi-Head Attention │  │
    │  │   (4 heads)          │  │
    │  └──────────┬───────────┘  │
    │             │               │
    │  ┌──────────▼───────────┐  │
    │  │  Layer Norm + Residual│ │
    │  └──────────┬───────────┘  │
    │             │               │
    │  ┌──────────▼───────────┐  │
    │  │  Feed Forward        │  │
    │  │  (512 hidden dim)    │  │
    │  └──────────┬───────────┘  │
    │             │               │
    │  ┌──────────▼───────────┐  │
    │  │  Layer Norm + Residual│ │
    │  └──────────┬───────────┘  │
    │             │               │
    │  Repeat 2-3 times           │
    └─────────────┬──────────────┘
                  │
         [Batch, 47, 32]
                  │
         ┌────────▼────────┐
         │ Extract [CLS]   │
         │ Token           │
         └────────┬────────┘
                  │
         [Batch, 32]
                  │
    ┌─────────────▼──────────────┐
    │  MLP Classification Head   │
    │  ┌──────────────────────┐  │
    │  │ Linear(32 → 128)     │  │
    │  │ ReLU + Dropout(0.1)  │  │
    │  │ Linear(128 → 34)     │  │
    │  └──────────┬───────────┘  │
    └─────────────┬──────────────┘
                  │
         [Batch, 34 classes]
```

## Component Details

### 1. Feature Embedding Module

```python
class FeatureEmbedding(nn.Module):
    """
    Handles different feature types:
    - Categorical: Learned embeddings
    - Numerical: Linear projection with normalization
    """
    
    def __init__(self, 
                 cat_feature_dims: List[int],  # Cardinality of each categorical feature
                 num_numerical_features: int,
                 embedding_dim: int = 32):
        
        # Categorical feature embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim, embedding_dim=embedding_dim)
            for dim in cat_feature_dims
        ])
        
        # Numerical feature projection
        self.num_projection = nn.Linear(num_numerical_features, embedding_dim)
        self.num_norm = nn.LayerNorm(embedding_dim)
```

**Key Decisions:**
- **Embedding Dim = 32**: Balance between expressivity and model size
- **Separate Processing**: Categorical and numerical features handled differently (matching TabTransformer paper)
- **Layer Normalization**: Stabilizes numerical feature magnitudes

### 2. Transformer Encoder

```python
class TransformerEncoder(nn.Module):
    def __init__(self,
                 embed_dim: int = 32,
                 num_heads: int = 4,
                 ff_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
```

**Key Decisions:**
- **Number of Heads = 4**: Sufficient for 46 features, keeps computation light
- **Number of Layers = 2**: Balance between model capacity and overfitting risk with limited data
- **FF Hidden Dim = 128**: 4x embedding dimension (standard Transformer ratio)
- **GELU Activation**: Better than ReLU for transformers (smoother gradients)

### 3. Classification Head

```python
class ClassificationHead(nn.Module):
    def __init__(self, 
                 embed_dim: int = 32,
                 hidden_dim: int = 128,
                 num_classes: int = 34,
                 dropout: float = 0.1):
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
```

## Federated Learning Integration

### Weight Aggregation (FedAvg)

```python
def federated_averaging(client_weights: List[Dict], 
                       client_sizes: List[int]) -> Dict:
    """
    Aggregate PyTorch state_dicts from multiple clients.
    
    Args:
        client_weights: List of model.state_dict() from each client
        client_sizes: Number of samples per client (for weighted averaging)
    
    Returns:
        Aggregated state_dict
    """
    total_size = sum(client_sizes)
    
    # Initialize with first client's structure
    avg_weights = {}
    
    for key in client_weights[0].keys():
        # Weighted average of each parameter
        avg_weights[key] = sum(
            (client_sizes[i] / total_size) * client_weights[i][key]
            for i in range(len(client_weights))
        )
    
    return avg_weights
```

**Key Points:**
- PyTorch `state_dict()` contains all model parameters as tensors
- Weighted averaging accounts for different client data sizes
- Same structure as TensorFlow version, just different API

### Client Training Loop

```python
def train_client(model, train_loader, epochs=5, lr=0.001, device='cpu'):
    """
    Train model locally on client data.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    return model.state_dict(), total_loss / len(train_loader)
```

## Feature Engineering for Network Attack Data

### Categorical vs Numerical Feature Split

Based on typical IoT network intrusion datasets (CIC-IoT-2023):

**Categorical Features (~20):**
- Protocol (TCP, UDP, ICMP, etc.)
- Source/Destination Port categories (well-known, registered, dynamic)
- TCP Flags (SYN, ACK, FIN, RST, etc.)
- Connection state
- Service type

**Numerical Features (~26):**
- Packet counts (forward/backward)
- Byte counts
- Packet length statistics (min, max, mean, std)
- Flow duration
- Inter-arrival times
- Window sizes
- Header lengths

### Preprocessing Pipeline

```python
class NetworkFeaturePreprocessor:
    """
    Handles feature preprocessing for TabTransformer.
    """
    
    def __init__(self, categorical_columns, numerical_columns):
        self.cat_cols = categorical_columns
        self.num_cols = numerical_columns
        
        # Categorical: Label encoding
        self.label_encoders = {}
        for col in categorical_columns:
            self.label_encoders[col] = LabelEncoder()
        
        # Numerical: Standardization (already done by existing scaler)
        self.scaler = None  # Use existing scaler from data_utils.py
    
    def fit_transform(self, df):
        # Encode categorical features
        cat_encoded = []
        for col in self.cat_cols:
            encoded = self.label_encoders[col].fit_transform(df[col])
            cat_encoded.append(encoded)
        
        # Scale numerical features (use existing scaler)
        num_scaled = self.scaler.fit_transform(df[self.num_cols])
        
        return np.array(cat_encoded).T, num_scaled
```

## Hyperparameters for FL Stability

### Critical FL-Specific Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Local Learning Rate** | 0.0005 | Lower than centralized (0.001) to prevent client drift |
| **Local Epochs** | 3-5 | Too many causes overfitting to local data |
| **Batch Size** | 128-256 | Larger batches = more stable gradients |
| **Communication Rounds** | 30-50 | More rounds compensates for fewer local epochs |
| **Client Fraction** | 1.0 (all 5) | Small number of clients, use all each round |
| **Warmup Rounds** | 5 | Gradually increase LR to stabilize early training |

### Model-Specific Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Embedding Dim** | 32 | Sufficient for 46 features, keeps model light |
| **Num Attention Heads** | 4 | Divides evenly into 32, standard choice |
| **Num Transformer Layers** | 2-3 | More layers = better performance but slower |
| **FF Hidden Dim** | 128 | 4x embedding dimension (Transformer standard) |
| **Dropout Rate** | 0.1-0.2 | Regularization, higher for more data |
| **Attention Dropout** | 0.1 | Prevents overfitting in attention weights |

### Learning Rate Schedule

```python
# Warmup + Cosine Annealing
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,
    epochs=local_epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,  # 30% warmup
    anneal_strategy='cos'
)
```

## Model Size and Efficiency

### Parameter Count Estimation

```
Feature Embeddings:
- 20 categorical features × 32 dim × avg 50 categories = ~32,000 params
- Numerical projection: 26 × 32 = 832 params
Subtotal: ~33K params

Transformer Encoder (per layer):
- Multi-head attention: (32×32×4) × 4 heads = 16,384 params
- FF network: (32×128) + (128×32) = 8,192 params
- Layer norms: negligible
Subtotal per layer: ~25K params
Total for 2 layers: ~50K params

Classification Head:
- Linear(32→128): 4,096 params
- Linear(128→34): 4,352 params
Subtotal: ~8.5K params

TOTAL: ~92,000 parameters (~370 KB in float32)
```

**Comparison:**
- Current DNN: 17,474 params (~70 KB)
- TabTransformer: 92,000 params  (~370 KB)
- Increase: 5.3x parameters, still very lightweight!

### Inference Time

```
Estimated on CPU (Intel i5):
- Embedding lookup: ~1ms
- Transformer forward (2 layers): ~5-10ms
- Classification head: ~1ms
Total: ~10-15ms per sample
Batch of 256: ~20-30ms

Acceptable for edge devices ✓
```

## File Structure Changes

```
Notebooks/
├── utils/
│   ├── model_utils.py          # Existing TensorFlow DNN
│   ├── model_utils_pytorch.py  # NEW: PyTorch TabTransformer
│   ├── fl_utils.py             # Existing TensorFlow FL
│   ├── fl_utils_pytorch.py     # NEW: PyTorch FL utilities
│   ├── feature_utils.py        # NEW: Feature preprocessing
│   └── data_utils.py           # MODIFIED: Add categorical/numerical split
│
├── configs/
│   └── training_config.yaml    # MODIFIED: Add TabTransformer config
│
├── 1_Data_Preprocessing.ipynb  # MODIFIED: Add feature type identification
├── 2_Federated_Training.ipynb  # MODIFIED: Support both TF and PyTorch
└── 3_Model_Evaluation_Export.ipynb  # MODIFIED: Eval PyTorch models
```

## Migration Strategy

### Phase 1: Side-by-Side Implementation
1. Keep existing TensorFlow code intact
2. Add new PyTorch modules alongside
3. Use config flag to switch between frameworks
4. Compare results before deprecating TensorFlow

### Phase 2: Testing & Validation
1. Unit tests for TabTransformer components
2. Integration test for FL training loop
3. Benchmark against DNN baseline
4. Validate on small dataset first (11 files)

### Phase 3: Full Deployment
1. Train with full dataset (169 files)
2. Compare final metrics
3. If TabTransformer > DNN: make it default
4. Document differences and migration guide

## Risk Mitigation

### Risk: TabTransformer doesn't improve accuracy

**Mitigation:**
- Benchmark on validation set before full training
- Keep DNN as fallback option
- Analyze attention weights to debug if needed

### Risk: PyTorch introduces bugs

**Mitigation:**
- Extensive unit testing
- Side-by-side comparison with TensorFlow
- Gradual rollout (small dataset → full dataset)

### Risk: FL convergence issues

**Mitigation:**
- Use proven hyperparameters from FL literature
- Implement learning rate warmup
- Monitor client drift metrics
- Add gradient clipping if needed

## Open Questions

1. **Exact Feature Types**: Need to inspect actual dataset to determine which of the 46 features are categorical vs numerical. This affects embedding layer design.

2. **Training Data**: With only 11 files currently showing 40.75% accuracy, should we proceed with TabTransformer immediately or wait for full dataset?

3. **Baseline Comparison**: Should we first re-run current DNN with full 169 files to establish a proper baseline?

4. **Production Timeline**: Is this for thesis deadline or production deployment? Affects testing thoroughness.
