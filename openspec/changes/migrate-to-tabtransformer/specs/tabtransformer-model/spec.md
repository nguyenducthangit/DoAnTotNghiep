# TabTransformer Model Architecture Specification

## ADDED Requirements

### Requirement: TabTransformer model shall process tabular network attack data using separate pathways for categorical and numerical features

The TabTransformer model SHALL process categorical features (e.g., protocol, port, flags) and numerical features (e.g., packet counts, durations) through separate pathways to leverage transformer-based architecture strengths on tabular data.

#### Scenario: Embedding categorical features

**Given** a batch of network traffic samples with 20 categorical features (protocol type, port categories, TCP flags, etc.)  
**When** the features are passed to the TabTransformer model  
**Then** each categorical feature is embedded into a 32-dimensional vector space using learned embeddings  
**And** the embeddings preserve semantic relationships between similar categorical values

#### Scenario: Projecting numerical features

**Given** a batch of network traffic samples with 26 numerical features (packet counts, byte counts, durations, etc.)  
**When** the features are passed to the TabTransformer model  
**Then** numerical features are projected into the same 32-dimensional space as categorical embeddings using a linear layer  
**And** numerical projections are normalized to match the scale of categorical embeddings

---

### Requirement: TabTransformer shall use multi-head self-attention to learn feature interactions

The TabTransformer SHALL employ multi-head self-attention through transformer encoder layers to capture complex relationships between network traffic features indicative of attack patterns.

#### Scenario: Attention-based feature interaction learning

**Given** embedded and projected features from all 46 input dimensions  
**When** features pass through 2 transformer encoder layers with 4 attention heads each  
**Then** the model learns weighted relationships between features (e.g., high attention between protocol + port + packet size)  
**And** attention weights can be extracted for interpretability

#### Scenario: Residual connections prevent degradation

**Given** a transformer encoder with multiple layers  
**When** features pass through the encoder  
**Then** residual connections are applied around attention and feed-forward sub-layers  
**And** gradient flow is maintained through deeper networks

---

### Requirement: TabTransformer shall output attack class predictions for 34 categories

The TabTransformer SHALL classify network traffic samples into one of 34 distinct attack categories or benign traffic.

#### Scenario: Multi-class classification output

**Given** encoded features from the transformer layers  
**When** features are passed through the classification head  
**Then** the model outputs a 34-dimensional logit vector  
**And** applying softmax yields class probabilities summing to 1.0  
**And** the predicted class is the argmax of the probability distribution

---

### Requirement: TabTransformer shall be lightweight enough for edge/IoT device deployment

The TabTransformer model SHALL maintain computational efficiency suitable for deployment on resource-constrained federated learning edge/IoT clients.

#### Scenario: Model size constraint

**Given** the TabTransformer architecture with specified hyperparameters  
**When** the model is instantiated and parameters are counted  
**Then** total parameter count is approximately 92,000 parameters  
**And** model file size is under 2MB in FP32 format  
**And** model file size is under 500KB when quantized to FP16

#### Scenario: Inference time constraint

**Given** a trained TabTransformer model running on CPU  
**When** a batch of 256 samples is passed through the model  
**Then** inference completes in under 50ms  
**And** single-sample inference completes in under 5ms

---

### Requirement: TabTransformer architecture shall be configurable via YAML

The TabTransformer architecture SHALL be configurable via YAML with all hyperparameters (embedding dimension, transformer layers, attention heads, dropout rates) specified in the training configuration file.

#### Scenario: Loading model configuration

**Given** a `training_config.yaml` with TabTransformer hyperparameters  
**When** the configuration is loaded  
**Then** model is instantiated with:
- embedding_dim from config (default: 32)
- num_transformer_layers from config (default: 2)
- num_attention_heads from config (default: 4)
- ff_hidden_dim from config (default: 128)
- dropout_rate from config (default: 0.1)

#### Scenario: Validating configuration compatibility

**Given** a TabTransformer configuration in YAML  
**When** the configuration is validated  
**Then** embedding_dim must be divisible by num_attention_heads  
**And** num_attention_heads must be >= 1  
**And** num_transformer_layers must be >= 1  
**And** dropout_rate must be in range [0.0, 0.5]
