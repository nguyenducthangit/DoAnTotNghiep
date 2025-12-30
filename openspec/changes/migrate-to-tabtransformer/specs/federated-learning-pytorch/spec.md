# Federated Learning with PyTorch Specification

## ADDED Requirements

### Requirement: FL system shall support PyTorch models with state_dict serialization

The federated learning system SHALL support PyTorch models using state_dict serialization for weight extraction, transmission, and aggregation.

#### Scenario: Extracting client model weights

**Given** a TabTransformer model trained locally on a client  
**When** the client completes local training  
**Then** model weights are extracted using `model.state_dict()`  
**And** the state_dict contains all model parameters as PyTorch tensors  
**And** the state_dict is serializable for transmission to the server

#### Scenario: Loading aggregated weights into client model

**Given** aggregated weights from the federated server  
**When** a client receives the global model update  
**Then** client loads weights using `model.load_state_dict(aggregated_weights)`  
**And** model parameters are updated to match the global model  
**And** model is ready for next round of local training

---

### Requirement: Server shall aggregate client models using Federated Averaging (FedAvg)

The federated server SHALL aggregate client model parameters using weighted averaging (FedAvg) with weights proportional to client dataset sizes.

#### Scenario: Weighted aggregation of multiple clients

**Given** 5 clients have completed local training with dataset sizes [1000, 1200, 800, 1500, 900]  
**When** the server performs FedAvg aggregation  
**Then** each parameter in the global model is computed as:  
  `global_param = sum(client_size_i / total_size * client_param_i for i in clients)`  
**And** clients with more data contribute proportionally more to the global model  
**And** all parameters in state_dict are aggregated independently

#### Scenario: Handling non-participating clients

**Given** 5 total clients but only 3 participate in a given round  
**When** the server performs aggregation  
**Then** only participating client weights are included in the average  
**And** total_size is the sum of participating client dataset sizes only  
**And** global model is updated based on available clients

---

### Requirement: FL training shall support configurable local training hyperparameters

The FL system SHALL support configurable local training hyperparameters (epochs, batch size, learning rate) for each client to balance convergence speed and communication efficiency.

#### Scenario: Local client training configuration

**Given** FL configuration with local_epochs=5, batch_size=256, learning_rate=0.0005  
**When** a client starts local training  
**Then** client trains for exactly 5 epochs  
**And** uses batches of 256 samples  
**And** Adam optimizer is initialized with lr=0.0005  
**And** training completes without exceeding memory constraints on edge devices

#### Scenario: Learning rate scheduling for FL stability

**Given** a client training with a learning rate scheduler  
**When** local training progresses through epochs  
**Then** learning rate is reduced after epochs with no loss improvement  
**And** minimum learning rate is maintained at 1e-5  
**And** scheduler state is reset between FL rounds (not accumulated)

---

### Requirement: FL system shall complete specified communication rounds

The FL training process SHALL execute the configured number of communication rounds where clients train locally and the server aggregates weights.

#### Scenario: Executing FL training rounds

**Given** configuration specifying 30 communication rounds  
**When** FL training starts  
**Then** the following occurs 30 times in sequence:
1. Server broadcasts current global model to all clients
2. Each client trains locally on private data
3. Clients send updated weights to server
4. Server aggregates weights using FedAvg
5. Global model is updated with aggregated weights  
**And** training history is logged for each round

#### Scenario: Early stopping based on convergence

**Given** FL training configured for 30 rounds with early stopping enabled  
**When** global model accuracy plateaus for 5 consecutive rounds  
**Then** training terminates early before 30 rounds  
**And** best model from highest accuracy round is saved  
**And** training history indicates early stopping was triggered

---

### Requirement: FL system shall evaluate global model on test set after each round

The FL system SHALL evaluate the global model on a held-out test set after each communication round to monitor training progress.

#### Scenario: Test set evaluation after aggregation

**Given** the server has aggregated client models for round N  
**When** aggregation is complete  
**Then** global model is evaluated on the full test set  
**And** evaluation metrics (loss, accuracy, per-class F1) are computed  
**And** metrics are logged to training history  
**And** metrics are displayed to show training progress

#### Scenario: Detecting accuracy degradation

**Given** global model accuracy at round N was 85%  
**When** round N+1 completes and accuracy drops to 80%  
**Then** warning is logged about potential divergence  
**And** training continues (unless early stopping triggered)  
**And** degradation is visible in training history charts

---

### Requirement: FL implementation shall maintain compatibility with existing TensorFlow code

The PyTorch FL implementation SHALL maintain backward compatibility with the existing TensorFlow DNN implementation to support gradual migration and provide fallback options.

#### Scenario: Framework selection via configuration

**Given** training configuration with `framework: 'pytorch'`  
**When** training notebook is executed  
**Then** PyTorch TabTransformer is instantiated  
**And** PyTorch training loop is used  
**And** PyTorch model is saved  

**Given** training configuration with `framework: 'tensorflow'`  
**When** training notebook is executed  
**Then** TensorFlow DNN is instantiated  
**And** TensorFlow/Keras training loop is used  
**And** Keras .h5 model is saved

#### Scenario: Side-by-side comparison

**Given** both TensorFlow and PyTorch models have been trained  
**When** evaluation notebook is run  
**Then** both models are loaded and evaluated on the same test set  
**And** comparison metrics (accuracy difference, F1-score difference) are computed  
**And** results are displayed in a comparison table
