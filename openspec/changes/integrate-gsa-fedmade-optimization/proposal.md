# Integrate GSA Feature Selection & FedMade Aggregation

## Why

The current Federated Learning system uses **all 46 features** from the CICIoT2023 dataset without optimization, and aggregates client models using standard **FedAvg** (simple averaging). According to `planproject.md`, the system should integrate:

1. **GSA (Gravitational Search Algorithm)** for intelligent feature selection during preprocessing
2. **FedMade** aggregation strategy for sophisticated model weight aggregation

These optimizations are critical for achieving the target **>95% accuracy** on IoT network attack detection with 34 attack classes.

### Current Limitations

1. **No Feature Optimization**: Using all 46 features may include noisy or redundant features that hurt model performance
2. **Simple Aggregation**: Standard FedAvg treats all client updates equally without considering data quality or contribution strength
3. **Suboptimal Accuracy**: Without feature selection and advanced aggregation, the model may not reach the 95% accuracy target

### Expected Benefits

- **Better Feature Quality**: GSA eliminates noisy/redundant features, keeping only the most discriminative ones
- **Improved Model Performance**: Cleaner feature set → better accuracy and faster convergence
- **Smarter Aggregation**: FedMade can handle heterogeneous client contributions more effectively than simple averaging
- **Research Contribution**: Integrating GSA + FedMade + TabTransformer represents a novel combination for IoT attack detection

## What Changes

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: Centralized Feature Selection (GSA)                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Load All Data → Clean → Run GSA → Select Best Features  │  │
│  │ Output: Optimized feature list (e.g., 15-25 from 46)     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: Data Partitioning (Post-GSA)                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Split Global Test (20%) → Partition Train for Clients    │  │
│  │ All partitions use ONLY GSA-selected features            │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: Federated Training (FedMade Aggregation)              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Server → Clients: Send Global Weights                    │  │
│  │ Clients: Local Training on Selected Features             │  │
│  │ Clients → Server: Send Updated Weights                   │  │
│  │ Server: Aggregate with FedMade (not simple average!)     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. GSA Feature Selection Module

**New File**: `Notebooks/utils/gsa_algorithm.py`

The GSA algorithm treats feature subsets as particles with mass in a search space:
- **Fitness Function**: Classification accuracy on validation set
- **Search Process**: Iteratively evaluate feature combinations
- **Output**: Binary vector indicating which features to keep

Key functions:
```python
def gsa_feature_selection(X, y, num_features_to_select, max_iterations, population_size)
def evaluate_feature_subset(X, y, feature_mask)
def calculate_gravitational_forces(population, fitness_scores)
```

#### 2. FedMade Aggregation Strategy

**New File**: `Notebooks/utils/fedmade_aggregation.py`

FedMade (Federated Model Aggregation with Dynamic Evaluation) uses sophisticated weight aggregation:
- **Client Contribution Scoring**: Evaluate each client's model quality
- **Layer-wise Matching**: Intelligently match and merge layer weights
- **Adaptive Weighting**: Dynamic weights based on client performance

Key functions:
```python
def fedmade_aggregate(client_state_dicts, client_metrics, global_model)
def compute_client_contribution_scores(client_metrics)
def layer_wise_weighted_aggregation(client_layers, contribution_scores)
```

#### 3. Modified Preprocessing Pipeline

**Modified File**: `Notebooks/1_Data_Preprocessing.ipynb`

Add GSA step between cleaning and partitioning:
```python
# New workflow:
1. Load and merge all CSV files
2. Clean data (existing)
3. **[NEW] Run GSA feature selection**
4. **[NEW] Filter dataset to selected features only**
5. Encode labels (existing)
6. Normalize features (existing)
7. Partition data for clients (existing)
```

#### 4. Modified FL Training Loop

**Modified Files**: 
- `Notebooks/2_Federated_Training.ipynb`
- `Notebooks/utils/fl_utils_pytorch.py`

Replace `federated_averaging_pytorch()` calls with `fedmade_aggregate()`:
```python
# Old:
aggregated_state_dict = federated_averaging_pytorch(client_state_dicts, client_sizes)

# New:
aggregated_state_dict = fedmade_aggregate(
    client_state_dicts, 
    client_metrics,  # Pass client validation metrics
    global_model
)
```

#### 5. Configuration Updates

**Modified File**: `Notebooks/configs/training_config.yaml`

Add new configuration sections:
```yaml
# GSA Feature Selection
gsa:
  enabled: true
  num_features_to_select: 20       # Target number of features
  max_iterations: 50               # GSA iterations
  population_size: 30              # Number of candidate solutions
  gravitational_constant: 100.0
  alpha: 20.0                      # Gravitational constant decay

# Aggregation Strategy
aggregation:
  method: fedmade                  # 'fedavg' or 'fedmade'
  fedmade:
    use_client_metrics: true       # Use validation accuracy for weighting
    layer_matching: true           # Enable layer-wise matching
    contribution_threshold: 0.3    # Min contribution score to include
```

### File Changes Summary

#### NEW Files
- `Notebooks/utils/gsa_algorithm.py` - GSA implementation
- `Notebooks/utils/fedmade_aggregation.py` - FedMade aggregation
- `Output/gsa_results/selected_features.json` - GSA output
- `Output/gsa_results/gsa_convergence.png` - GSA visualization

#### MODIFIED Files
- `Notebooks/1_Data_Preprocessing.ipynb` - Add GSA step
- `Notebooks/2_Federated_Training.ipynb` - Use FedMade aggregation
- `Notebooks/utils/fl_utils_pytorch.py` - Import and use FedMade
- `Notebooks/configs/training_config.yaml` - Add GSA and FedMade configs
- `Notebooks/utils/data_utils.py` - Support feature filtering

## Impact

### Affected Specs
- **NEW**: `specs/preprocessing-optimization/spec.md` - GSA feature selection requirements
- **NEW**: `specs/advanced-aggregation/spec.md` - FedMade aggregation requirements

### Affected Code
- **NEW**: 2 new utility modules (gsa_algorithm.py, fedmade_aggregation.py)
- **MODIFIED**: 3 notebooks, 2 utility files, 1 config file

### Dependencies
- **Python Packages**: No new dependencies required (uses numpy, sklearn)
- **Computational**: GSA requires 30-60 minutes on full dataset
- **Sequence**: GSA must complete before data partitioning

### Success Criteria

#### GSA Feature Selection
✅ GSA successfully converges within configured iterations  
✅ Selected features reduce dimensionality by 30-50% (e.g., 46 → 20-25 features)  
✅ Selected feature set achieves ≥95% accuracy on validation set  
✅ Feature selection results are reproducible (same seed → same features)  
✅ GSA convergence curve shows fitness improvement over iterations

#### FedMade Aggregation
✅ FedMade aggregation produces valid model state_dict  
✅ Aggregated model maintains or improves accuracy vs FedAvg  
✅ Client contribution scores correctly reflect performance differences  
✅ Training converges to ≥95% accuracy on test set  
✅ FedMade shows faster convergence than FedAvg (fewer rounds to target accuracy)

#### Integration
✅ All notebooks execute without errors in sequence (1→2→3)  
✅ Configuration toggle works: can switch between FedAvg/FedMade  
✅ Output artifacts include GSA results and aggregation metrics  
✅ Model with GSA+FedMade outperforms baseline (all features + FedAvg)

## User Review Required

> [!IMPORTANT]
> **Research Implementation**: FedMade is described in `planproject.md` as potentially using "layer-matching" or "dynamic weighting." Do you have a specific paper or reference for the FedMade algorithm? This will ensure accurate implementation.

> [!WARNING]
> **Computational Cost**: GSA requires evaluating many feature subsets. With 46 features and ~12GB data, this could take 30-60 minutes even with sampling. Consider running GSA on a sampled subset (10-20% of data) if full dataset is too slow.

> [!IMPORTANT]
> **Feature Count**: The plan suggests selecting 15-25 features from 46. Do you have a specific target number, or should GSA automatically determine the optimal count based on a performance threshold?

## Notes

### GSA Algorithm Details

The Gravitational Search Algorithm is a nature-inspired metaheuristic:
1. **Initialization**: Random feature subsets (binary vectors)
2. **Fitness Evaluation**: Train simple model (e.g., Random Forest) and measure accuracy
3. **Force Calculation**: Better solutions (higher accuracy) exert stronger gravitational pull
4. **Movement**: Worse solutions move toward better ones in feature space
5. **Iteration**: Repeat until convergence or max iterations

### FedMade vs FedAvg

| Aspect | FedAvg (Current) | FedMade (Proposed) |
|--------|------------------|-------------------|
| Weighting | Equal or by sample count | By validation performance |
| Layer Treatment | Uniform averaging | Layer-wise intelligent matching |
| Client Selection | All included | Can exclude low-quality clients |
| Convergence Speed | Baseline | Potentially faster |
| Complexity | O(n) simple average | O(n·m) where m = layers |

### Implementation Timeline

- **Phase 1 - GSA**: 2-3 days (algorithm + integration)
- **Phase 2 - FedMade**: 2-3 days (implementation + testing)
- **Phase 3 - Integration**: 1-2 days (notebooks + validation)
- **Phase 4 - Evaluation**: 1-2 days (benchmarking + documentation)
- **Total**: ~7-10 days

### Reference Questions

Before starting implementation, please clarify:

1. **FedMade Specification**: Do you have a paper/algorithm description for FedMade, or should I implement a standard dynamic weighted aggregation strategy?

2. **GSA Target**: Fixed number of features (e.g., 20) or performance-based (e.g., keep adding features until 95% accuracy)?

3. **Computational Budget**: Is 30-60 minutes acceptable for GSA, or should I use sampling/parallelization?

4. **Validation Strategy**: Should GSA use k-fold cross-validation or a simple train/val split for fitness evaluation?
