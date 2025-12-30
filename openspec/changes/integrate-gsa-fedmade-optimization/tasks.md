# Implementation Tasks: GSA + FedMade Integration

## Phase 1: GSA Feature Selection Implementation

### 1.1 Create GSA Algorithm Module
- [ ] Create `Notebooks/utils/gsa_algorithm.py`
- [ ] Implement `GravitationalSearchAlgorithm` class
  - [ ] `__init__`: Initialize population, parameters
  - [ ] `calculate_fitness`: Evaluate feature subset quality
  - [ ] `calculate_masses`: Compute particle masses based on fitness
  - [ ] `calculate_forces`: Compute gravitational forces
  - [ ] `update_velocities`: Update particle velocities
  - [ ] `update_positions`: Update particle positions (binary encoding)
  - [ ] `run`: Main GSA loop
- [ ] Implement helper functions
  - [ ] `evaluate_feature_subset`: Train classifier and measure accuracy
  - [ ] `binary_sigmoid`: Convert continuous to binary positions
  - [ ] `plot_convergence`: Visualize GSA convergence curve
- [ ] Add comprehensive docstrings and type hints
- [ ] Validate: Test on simple dataset (iris/digits)

### 1.2 Integrate GSA into Preprocessing Notebook
- [ ] Modify `Notebooks/1_Data_Preprocessing.ipynb`
  - [ ] Add GSA import section
  - [ ] Add new cell: "Feature Selection with GSA"
  - [ ] Load merged & cleaned data
  - [ ] Configure GSA parameters from config
  - [ ] Run GSA feature selection
  - [ ] Save selected features to JSON
  - [ ] Filter dataset to selected features only
  - [ ] Update subsequent cells to use filtered features
  - [ ] Generate and save GSA convergence plot
- [ ] Validate: Run notebook end-to-end on sample data

### 1.3 Update Data Utilities for Feature Filtering
- [ ] Modify `Notebooks/utils/data_utils.py`
  - [ ] Add `filter_features_by_names` function
  - [ ] Add `load_selected_features` function
  - [ ] Modify `normalize_features` to accept feature list
  - [ ] Update `X_columns` handling to support dynamic feature lists
- [ ] Validate: Unit tests for feature filtering

## Phase 2: FedMade Aggregation Implementation

### 2.1 Create FedMade Aggregation Module
- [ ] Create `Notebooks/utils/fedmade_aggregation.py`
- [ ] Implement core FedMade functions
  - [ ] `fedmade_aggregate`: Main aggregation function
  - [ ] `compute_client_contribution_scores`: Calculate client weights
  - [ ] `layer_wise_weighted_aggregation`: Aggregate each layer separately
  - [ ] `match_layer_dimensions`: Handle dimension mismatches
  - [ ] `filter_low_contribution_clients`: Remove poor clients
- [ ] Implement evaluation utilities
  - [ ] `evaluate_client_model`: Get validation metrics per client
  - [ ] `normalize_contribution_scores`: Normalize scores to sum to 1
- [ ] Add comprehensive docstrings and type hints
- [ ] Validate: Test with mock state_dicts

### 2.2 Integrate FedMade into FL Utilities
- [ ] Modify `Notebooks/utils/fl_utils_pytorch.py`
  - [ ] Import FedMade functions
  - [ ] Add `aggregation_method` parameter to `federated_training_loop_pytorch`
  - [ ] Add conditional: if method=='fedmade' use FedMade, else FedAvg
  - [ ] Track client validation metrics during training
  - [ ] Pass metrics to FedMade aggregation
  - [ ] Add logging for contribution scores
- [ ] Validate: Test both FedAvg and FedMade paths

### 2.3 Update FL Training Notebook
- [ ] Modify `Notebooks/2_Federated_Training.ipynb`
  - [ ] Add cell: "Configure Aggregation Strategy"
  - [ ] Load aggregation method from config
  - [ ] Pass aggregation method to training loop
  - [ ] Add cell: "Visualize Client Contributions" (for FedMade)
  - [ ] Plot contribution scores over rounds
  - [ ] Compare FedAvg vs FedMade convergence
- [ ] Validate: Run training with both strategies

## Phase 3: Configuration & Integration

### 3.1 Update Configuration File
- [ ] Modify `Notebooks/configs/training_config.yaml`
  - [ ] Add `gsa:` section with all parameters
  - [ ] Add `aggregation:` section with strategy selection
  - [ ] Add `fedmade:` subsection with FedMade parameters
  - [ ] Document all new parameters with comments
- [ ] Validate: YAML syntax check

### 3.2 Update Includes/Constants
- [ ] Modify `Notebooks/includes.py` (if needed)
  - [ ] Add `SELECTED_FEATURES_PATH` constant
  - [ ] Add `GSA_RESULTS_DIR` constant
- [ ] Validate: Import check

### 3.3 Create Output Directories
- [ ] Create `Output/gsa_results/` directory
- [ ] Create `Output/aggregation_metrics/` directory
- [ ] Add `.gitkeep` files to preserve structure

## Phase 4: Testing & Validation

### 4.1 Unit Tests
- [ ] Create `Notebooks/utils/test_gsa_algorithm.py`
  - [ ] Test GSA initialization
  - [ ] Test fitness calculation
  - [ ] Test convergence on toy problem
  - [ ] Test binary encoding/decoding
- [ ] Create `Notebooks/utils/test_fedmade_aggregation.py`
  - [ ] Test contribution score calculation
  - [ ] Test layer-wise aggregation
  - [ ] Test client filtering
  - [ ] Test equivalence to FedAvg when scores are equal

### 4.2 Integration Tests
- [ ] Test full pipeline on small dataset (11-file subset)
  - [ ] Run Notebook 1: Preprocessing with GSA
  - [ ] Verify selected features saved correctly
  - [ ] Run Notebook 2: Training with FedMade
  - [ ] Verify model converges
  - [ ] Run Notebook 3: Evaluation
  - [ ] Verify all outputs generated
- [ ] Compare results: Baseline vs GSA+FedMade
  - [ ] Baseline: All features + FedAvg
  - [ ] Optimized: GSA features + FedMade
  - [ ] Metrics: Accuracy, F1-score, training time

### 4.3 Performance Validation
- [ ] Run GSA on full dataset (169 files)
  - [ ] Measure GSA execution time
  - [ ] Verify feature reduction (30-50%)
  - [ ] Check selected features make semantic sense
- [ ] Run FL training with FedMade on full dataset
  - [ ] Target: ≥95% accuracy
  - [ ] Compare convergence speed vs FedAvg
  - [ ] Analyze contribution score distribution
- [ ] Generate comparison report
  - [ ] Accuracy comparison table
  - [ ] Convergence comparison plot
  - [ ] Feature importance visualization

## Phase 5: Documentation & Finalization

### 5.1 Update Documentation
- [ ] Update `Notebooks/README.md`
  - [ ] Document GSA feature selection process
  - [ ] Document FedMade aggregation strategy
  - [ ] Add usage examples
  - [ ] Update workflow diagrams
- [ ] Add inline documentation to notebooks
  - [ ] Explain GSA parameters
  - [ ] Explain FedMade parameters
  - [ ] Add interpretation notes

### 5.2 Create Artifacts
- [ ] Generate GSA results visualization
  - [ ] Convergence curve
  - [ ] Selected features list
  - [ ] Feature importance ranking
- [ ] Generate FedMade analysis
  - [ ] Client contribution heatmap
  - [ ] Aggregation weight evolution
  - [ ] Performance comparison charts

### 5.3 Final Validation
- [ ] Run complete pipeline end-to-end
- [ ] Verify all success criteria met
- [ ] Generate final performance report
- [ ] Archive baseline results for comparison

## Verification Checklist

### GSA Verification
✅ GSA converges within configured iterations  
✅ Selected features: 15-25 from original 46  
✅ Validation accuracy with selected features ≥95%  
✅ GSA is reproducible (same seed → same output)  
✅ Execution time acceptable (<60 minutes on full data)

### FedMade Verification
✅ FedMade produces valid model state_dict  
✅ Aggregated model maintains architecture consistency  
✅ Contribution scores are reasonable (high-performing clients get higher scores)  
✅ Training converges to ≥95% test accuracy  
✅ FedMade converges faster than FedAvg (fewer rounds or better accuracy)

### Integration Verification
✅ All notebooks execute sequentially without errors  
✅ Configuration switches work (FedAvg ↔ FedMade)  
✅ All output files generated correctly  
✅ Model can be loaded and used for inference  
✅ Performance improvement demonstrated over baseline

## Dependencies

- **No new Python packages required**: GSA and FedMade use numpy, torch, sklearn
- **Data requirement**: CICIoT2023 dataset in `DataSets/` directory
- **Compute requirement**: GPU recommended for faster GSA fitness evaluation
- **Time requirement**: ~2-3 hours for full pipeline with full dataset

## Notes

- Implement GSA first, as FedMade depends on having optimized features
- Test each phase on small dataset before running on full 169-file dataset
- Keep FedAvg as fallback option via configuration
- Save all intermediate results for comparison and thesis documentation
