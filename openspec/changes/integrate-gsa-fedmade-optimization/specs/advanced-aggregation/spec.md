# Spec: Advanced Aggregation (FedMade)

## Overview

This specification defines the requirements for implementing FedMade (Federated Model Aggregation with Dynamic Evaluation) as an advanced alternative to standard FedAvg aggregation in the federated learning system.

## ADDED Requirements

### Requirement: FedMade Aggregation Implementation

The system MUST implement FedMade aggregation that uses client performance metrics to compute dynamic contribution scores for weighted model aggregation.

#### Scenario: Compute Client Contribution Scores

**Given** client validation metrics including accuracy and loss  
**When** the server prepares to aggregate client models  
**Then** the system should:
- Extract accuracy and loss from each client's metrics
- Normalize loss values (invert so lower loss = higher score)
- Compute weighted combination: `score = 0.7 * accuracy + 0.3 * (1 - normalized_loss)`
- Normalize scores to sum to 1.0

**And** clients with higher validation accuracy should receive higher scores  
**And** scores should be in the range [0, 1]

#### Scenario: Layer-wise Weighted Aggregation

**Given** client model state_dicts and contribution scores  
**When** FedMade performs aggregation  
**Then** the system should:
- Iterate through each layer in the model
- For each layer, compute weighted average using contribution scores
- Ensure all clients have matching layer shapes

**And** the aggregated state_dict should have the same structure as client state_dicts

#### Scenario: Filter Low-Quality Clients

**Given** a contribution threshold of 0.3  
**And** clients with scores [0.35, 0.25, 0.40, 0.20, 0.30]  
**When** FedMade filters clients  
**Then** the system should:
- Exclude clients with scores < 0.3 (clients 1, 3)
- Include only clients [0, 2, 4] in aggregation
- Re-normalize scores among remaining clients

**And** if all clients are filtered out, fall back to standard FedAvg

#### Scenario: Fallback to FedAvg

**Given** FedMade aggregation encounters an error (e.g., all clients filtered)  
**When** the aggregation step executes  
**Then** the system should:
- Log a warning about using fallback
- Use standard FedAvg with sample-count weighting
- Return a valid aggregated state_dict

**And** training should continue without interruption

### Requirement: Client Validation Metrics Collection

The system MUST collect and track client validation metrics during federated training.

#### Scenario: Evaluate Client Model Locally

**Given** a client model after local training  
**And** a validation split of the client's data (20% of local data)  
**When** the client completes local training  
**Then** the system should:
- Evaluate the model on the validation split
- Compute validation accuracy
- Compute validation loss
- Return metrics dictionary with {'accuracy', 'loss', 'num_samples'}

**And** metrics should be sent to the server along with model weights

#### Scenario: Track Metrics Across Rounds

**Given** multiple FL training rounds  
**When** clients report validation metrics each round  
**Then** the system should:
- Store metrics history for each client
- Track metric evolution over rounds
- Save metrics to `Output/aggregation_metrics/client_metrics.json`

### Requirement: FedMade Configuration

The system MUST support configurable FedMade parameters through the training configuration file.

#### Scenario: Load Aggregation Configuration

**Given** a training_config.yaml file with aggregation section  
**When** the FL training notebook loads configuration  
**Then** the system should read:
- `aggregation.method`: 'fedavg' or 'fedmade'
- `aggregation.fedmade.use_client_metrics`: Boolean
- `aggregation.fedmade.layer_matching`: Boolean (enable layer-wise aggregation)
- `aggregation.fedmade.contribution_threshold`: Minimum score to include client
- `aggregation.fedmade.accuracy_weight`: Weight for accuracy in scoring (default 0.7)
- `aggregation.fedmade.loss_weight`: Weight for loss in scoring (default 0.3)

#### Scenario: Switch Between FedAvg and FedMade

**Given** a configuration with `aggregation.method = 'fedavg'`  
**When** FL training runs  
**Then** the system should use standard federated averaging  
**And** not compute contribution scores

**Given** a configuration with `aggregation.method = 'fedmade'`  
**When** FL training runs  
**Then** the system should use FedMade aggregation  
**And** compute and apply contribution scores

### Requirement: FedMade Performance Tracking

The system MUST track and visualize FedMade-specific metrics.

#### Scenario: Log Contribution Scores

**Given** an FL training round using FedMade  
**When** aggregation completes  
**Then** the system should log:
- Contribution score for each client
- List of filtered clients (if any)
- Aggregation method used

**And** logs should be saved to `Output/aggregation_metrics/contribution_scores.json`

#### Scenario: Visualize Client Contributions

**Given** completed FL training with FedMade  
**When** generating training report  
**Then** the system should create:
- A heatmap showing contribution scores per client per round
- A line plot showing score evolution over rounds
- A bar chart comparing final contribution scores

**And** visualizations should be saved to `Output/aggregation_metrics/`

### Requirement: FedMade Integration with FL Loop

FedMade MUST integrate seamlessly with the existing FL training loop.

#### Scenario: FedMade in FL Training Loop

**Given** an FL training round  
**When** the aggregation step executes  
**Then** the workflow should be:
1. Clients train locally and compute validation metrics
2. Clients send {state_dict, metrics} to server
3. Server collects all client submissions
4. **[FedMade Step]** Server computes contribution scores from metrics
5. **[FedMade Step]** Server filters low-quality clients if threshold set
6. **[FedMade Step]** Server performs weighted aggregation using scores
7. Server updates global model with aggregated weights
8. Server broadcasts updated model to clients

**And** the global model should converge to ≥95% test accuracy

#### Scenario: Handle Missing Metrics

**Given** a client that fails to report validation metrics  
**When** FedMade attempts aggregation  
**Then** the system should:
- Detect missing metrics
- Assign default score (1/num_clients) to that client
- Log a warning about missing metrics
- Continue aggregation with available metrics

## MODIFIED Requirements

### Requirement: FL Training Loop (Modified)

The existing FL training loop MUST be updated to support multiple aggregation strategies.

#### Scenario: Parameterized Aggregation

**Given** the `federated_training_loop_pytorch` function  
**When** called with `aggregation_method` parameter  
**Then** the function should:
- Accept 'fedavg' or 'fedmade' as valid values
- Route to the appropriate aggregation function
- Pass necessary parameters (metrics for FedMade, sample counts for FedAvg)

**And** the function signature should be backward compatible (default to 'fedavg')

## Success Criteria

✅ FedMade correctly computes contribution scores based on client metrics  
✅ Layer-wise aggregation produces valid model state_dict  
✅ Client filtering works correctly with configurable threshold  
✅ FedMade shows equal or better convergence compared to FedAvg  
✅ System achieves ≥95% test accuracy with FedMade  
✅ Contribution scores are tracked and visualized correctly  
✅ Configuration toggle between FedAvg/FedMade works seamlessly  
✅ Fallback to FedAvg works when FedMade encounters errors
