# Migrate to TabTransformer Architecture

## Why

The current DNN achieves only 40.75% accuracy with 11-file dataset, far below the 95% target. TabTransformer is specifically designed for tabular data and has demonstrated superior performance on attack classification tasks in published research. By using separate embeddings for categorical features and self-attention for feature interactions, TabTransformer can better learn the complex patterns in network traffic data that distinguish attack types. Moving to PyTorch also aligns with modern FL frameworks (Flower, PySyft) and enables easier implementation of advanced transformer architectures.

## Problem Statement

The current Federated Learning system uses a basic Deep Neural Network (DNN) with sequential Dense layers for IoT network attack detection on tabular data. While functional, this architecture has limitations:

1. **Suboptimal Feature Learning**: Standard DNNs treat all features uniformly without distinguishing between categorical and numerical features
2. **Limited Feature Interactions**: Dense layers don't explicitly model complex feature interactions that are crucial for attack pattern detection
3. **Scalability Issues**: Current model achieves only 40.75% accuracy with limited data, requiring significant data volume to reach 95% target

## Proposed Solution

Migrate from DNN to **TabTransformer**, a state-of-the-art architecture specifically designed for tabular data that:

- Uses **embedding layers** for categorical features (e.g., protocol types, port numbers)
- Uses **column-wise attention** to learn feature interactions
- Employs **Transformer encoders** to capture complex patterns
- Maintains compatibility with Federated Learning through consistent architecture across clients

## Why TabTransformer Over DNN?

### Advantages for Tabular Network Attack Data

1. **Explicit Categorical Handling**: Network traffic data contains many categorical features (protocols, flags, attack types). TabTransformer embeds these into rich representations.

2. **Self-Attention for Feature Interactions**: Multi-head attention can learn which features are most relevant together (e.g., port + protocol + packet size patterns).

3. **Better Data Efficiency**: TabTransformer typically requires less data to achieve high accuracy compared to standard DNNs on tabular data.

4. **State-of-the-Art Performance**: Research shows TabTransformer outperforms DNNs on tabular classification tasks, especially with moderate data volumes.

### Key Architectural Differences

| Aspect | DNN (Current) | TabTransformer (Proposed) |
|--------|---------------|---------------------------|
| Feature Processing | All features → Dense layers | Categorical → Embeddings, Numerical → Linear |
| Feature Interactions | Implicit through dense connections | Explicit through self-attention |
| Inductive Bias | None (generic) | Tabular-specific (column-wise) |
| Data Efficiency | Requires large datasets | Better with moderate datasets |
| Interpretability | Low | Medium (attention weights) |

## High-Level Architecture

```
Input Features (46 features)
     │
     ├─→ Categorical Features (e.g., protocol, flags)
     │        ↓
     │   Embedding Layers (dim=32 each)
     │        ↓
     │   [B, num_cat, embed_dim]
     │
     └─→ Numerical Features (e.g., packet sizes, counts)
              ↓
         Linear Projection (dim=32)
              ↓
         [B, num_num, embed_dim]
     
     Concatenate → [B, total_features, embed_dim]
              ↓
     Transformer Encoder Blocks (N layers)
         - Multi-Head Self-Attention
         - Feed-Forward Network
         - Layer Normalization
         - Residual Connections
              ↓
     [B, total_features, embed_dim]
              ↓
     Global Pooling (mean/cls token)
              ↓
     [B, embed_dim]
              ↓
     MLP Head (classification)
              ↓
     Output [B, 34 classes]
```

## Federated Learning Compatibility

### Model Consistency
- All clients use identical TabTransformer architecture
- Same embedding dimensions and number of transformer layers
- Deterministic initialization ensures weight shape consistency

### FedAvg Aggregation
- PyTorch models support `.state_dict()` for weight extraction
- Server aggregates client state dicts element-wise
- No changes needed to core FedAvg algorithm

### Edge Device Constraints
- Lightweight configuration: 2-3 Transformer layers, 4 attention heads
- Estimated parameters: ~500K (vs ~17K for current DNN, but much better performance)
- Model size: ~2MB (acceptable for edge devices)

## Migration from TensorFlow to PyTorch

### Why PyTorch?

1. **Better FL Support**: Industry-standard FL frameworks (Flower, PySyft) primarily support PyTorch
2. **Easier Custom Architectures**: TabTransformer requires custom attention mechanisms
3. **Better Debugging**: Dynamic computation graphs
4. **Research Community**: Most TabTransformer implementations in PyTorch

### Migration Strategy

1. Create new `model_utils_pytorch.py` alongside existing `model_utils.py`
2. Implement PyTorch versions of FL utilities in `fl_utils_pytorch.py`
3. Add configuration toggle to switch between TensorFlow/PyTorch
4. Gradually phase out TensorFlow after validation

## Implementation Scope

### In Scope
- ✅ TabTransformer model architecture (PyTorch)
- ✅ Feature preprocessing pipeline (categorical/numerical split)
- ✅ PyTorch training loop for FL clients
- ✅ FedAvg aggregation adapted for PyTorch
- ✅ Configuration updates to support new architecture
- ✅ Notebooks updated to use TabTransformer
- ✅ Documentation and migration guide

### Out of Scope
- ❌ Advanced FL strategies (FedProx, FedOpt) - future work
- ❌ Model compression/quantization - future work
- ❌ Distributed training across multiple GPUs - not needed for this scale
- ❌ Privacy mechanisms (differential privacy) - separate concern

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| TabTransformer requires more compute | Medium | Use lightweight config (2 layers, 4 heads) |
| PyTorch migration introduces bugs | High | Thorough testing, side-by-side comparison |
| Model convergence issues in FL | High | Careful hyperparameter tuning, learning rate scheduling |
| Accuracy doesn't improve over DNN | Medium | Benchmark both, keep DNN as fallback |

## Success Criteria

1. **Functional**: TabTransformer trains successfully in FL setup
2. **Performance**: Achieves ≥90% accuracy with 11-file dataset (vs current 40.75%)
3. **Performance**: Achieves ≥95% accuracy with full 169-file dataset
4. **Compatibility**: Works seamlessly with existing FL infrastructure
5. **Resource**: Model size ≤5MB, inference time ≤100ms per batch

## Timeline Estimate

- **Design & Proposal**: 1 day (current)
- **Implementation**: 3-4 days
- **Testing & Validation**: 2-3 days
- **Documentation**: 1 day
- **Total**: ~7-9 days

## User Review Required

> [!IMPORTANT]
> **Framework Migration**: This proposal switches from TensorFlow/Keras to PyTorch. While this provides better FL support and TabTransformer implementation, it requires rewriting the training infrastructure.

> [!WARNING]
> **Model Size Increase**: TabTransformer will be ~30x larger than current DNN (~2MB vs ~70KB). While still suitable for edge devices, this should be considered.

## Questions for Clarification

1. **Data Features**: Do you have a list of which of the 46 features are categorical vs numerical? This affects embedding design.

2. **Hardware Constraints**: What are the exact memory/compute limits for edge clients? (helps size the model)

3. **Preference on Migration**: Should we keep TensorFlow DNN as fallback, or fully commit to PyTorch?

4. **Training Time**: Is longer training time acceptable if accuracy improves significantly?
