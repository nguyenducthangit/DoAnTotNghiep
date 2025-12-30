# Implementation Tasks: TabTransformer Migration

## Pre-Implementation

- [ ] **T0.1**: Analyze dataset to identify categorical vs numerical features
  - Review CIC-IoT-2023 dataset documentation
  - Inspect actual CSV files to determine feature types
  - Create feature type mapping configuration
  - **Validation**: Feature types documented in `configs/feature_config.yaml`

- [ ] **T0.2**: Establish DNN baseline with full dataset (169 files)
  - Re-run existing DNN with 30 rounds on full data
  - Document accuracy, F1-scores, and training time
  - **Validation**: Baseline results saved in `Output/metrics/dnn_baseline.json`

## Phase 1: Core TabTransformer Implementation

- [ ] **T1.1**: Create `model_utils_pytorch.py` with base TabTransformer class
  - Implement `FeatureEmbedding` module
  - Implement `TransformerEncoder` module
  - Implement `ClassificationHead` module
  - Implement `TabTransformer` main class
  - **Validation**: `python Notebooks/utils/model_utils_pytorch.py` runs without errors

- [ ] **T1.2**: Add unit tests for TabTransformer components
  - Test feature embedding with sample data
  - Test transformer forward pass
  - Test classification head output shape
  - Test end-to-end model with dummy data
  - **Validation**: `pytest Notebooks/tests/test_tabtransformer.py` passes

- [ ] **T1.3**: Create `feature_utils.py` for preprocessing
  - Implement categorical feature encoding
  - Implement numerical feature normalization
  - Implement feature type detector
  - Add data loader factory for PyTorch
  - **Validation**: Features correctly preprocessed for sample batch

## Phase 2: FL Integration

- [ ] **T2.1**: Create `fl_utils_pytorch.py` for PyTorch FL
  - Implement client training function
  - Implement FedAvg aggregation function
  - Implement server broadcast function
  - Implement evaluation function
  - **Validation**: Unit test FL functions with mock clients

- [ ] **T2.2**: Add FL-specific utilities
  - Implement learning rate scheduler (warmup + cosine)
  - Add gradient clipping utilities
  - Implement client drift monitoring
  - Add checkpoint saving/loading
  - **Validation**: Utilities work with sample training loop

- [ ] **T2.3**: Update configuration system
  - Add TabTransformer hyperparameters to `training_config.yaml`
  - Add framework selection flag (tensorflow/pytorch)
  - Add feature type configuration
  - Add FL-specific hyperparameters
  - **Validation**: Config loads correctly with both frameworks

## Phase 3: Data Pipeline Updates

- [ ] **T3.1**: Modify `data_utils.py` for feature type handling
  - Add function to split categorical/numerical features
  - Update data loading to support PyTorch tensors
  - Implement stratified partitioning with feature types
  - **Validation**: Data loads correctly for both TF and PyTorch

- [ ] **T3.2**: Update `1_Data_Preprocessing.ipynb`
  - Add feature type identification cell
  - Add categorical feature cardinality analysis
  - Save feature metadata for TabTransformer
  - **Validation**: Notebook runs end-to-end, saves feature config

## Phase 4: Training Infrastructure

- [ ] **T4.1**: Update `2_Federated_Training.ipynb` for PyTorch
  - Add framework selection cell
  - Implement PyTorch training loop for TabTransformer
  - Add TabTransformer-specific monitoring (attention weights)
  - Maintain backward compatibility with TensorFlow DNN
  - **Validation**: Notebook trains TabTransformer for 5 rounds successfully

- [ ] **T4.2**: Add training utilities
  - Implement early stopping
  - Add model checkpointing
  - Implement training history logging
  - Add tensorboard logging (optional)
  - **Validation**: Training runs with all utilities enabled

## Phase 5: Evaluation & Comparison

- [ ] **T5.1**: Update `3_Model_Evaluation_Export.ipynb`
  - Add PyTorch model loading
  - Add TabTransformer evaluation metrics
  - Implement attention weight visualization
  - Add DNN vs TabTransformer comparison
  - **Validation**: Notebook evaluates TabTransformer model

- [ ] **T5.2**: Create comparison notebook
  - Side-by-side DNN vs TabTransformer results
  - Accuracy comparison across 34 classes
  - Training time and resource usage comparison
  - Feature importance from attention weights
  - **Validation**: Comprehensive comparison report generated

## Phase 6: Testing & Validation

- [ ] **T6.1**: Small dataset validation (11 files)
  - Train TabTransformer with 11 files for 10 rounds
  - Compare with DNN baseline (40.75% accuracy)
  - Verify model convergence (loss should decrease)
  - Check for memory/compute issues
  - **Target**: Accuracy > 50% (better than DNN baseline)
  - **Validation**: Results documented in `Output/metrics/tabtransformer_small.json`

- [ ] **T6.2**: Full dataset validation (169 files)
  - Train TabTransformer with full dataset for 30 rounds
  - Monitor convergence over rounds
  - Evaluate on test set
  - **Target**: Accuracy ≥ 95%, all classes F1 ≥ 0.85
  - **Validation**: Results meet target metrics

- [ ] **T6.3**: Integration tests
  - Test FL workflow end-to-end
  - Test model save/load
  - Test with different numbers of clients (3, 5, 7)
  - Test with different data distributions
  - **Validation**: All integration tests pass

## Phase 7: Documentation

- [ ] **T7.1**: Create migration guide
  - Document differences between DNN and TabTransformer
  - Provide step-by-step migration instructions
  - Document new hyperparameters
  - Add troubleshooting section
  - **Validation**: Guide reviewed and approved

- [ ] **T7.2**: Update code documentation
  - Add docstrings to all new functions
  - Update README with TabTransformer instructions
  - Create architecture diagram
  - Document feature type requirements
  - **Validation**: All code has proper documentation

- [ ] **T7.3**: Create performance report
  - Benchmark DNN vs TabTransformer
  - Document accuracy improvements
  - Document resource usage (memory, time)
  - Provide recommendations for production use
  - **Validation**: Report approved for thesis/production

## Phase 8: Cleanup & Finalization

- [ ] **T8.1**: Code review and refactoring
  - Remove dead code
  - Optimize performance bottlenecks
  - Ensure consistent code style
  - Add type hints where missing
  - **Validation**: Code passes linting (black, flake8, mypy)

- [ ] **T8.2**: Final validation
  - Run all tests one final time
  - Verify all notebooks work end-to-end
  - Check model outputs are reproducible
  - Validate configuration files
  - **Validation**: All tests pass, notebooks run successfully

- [ ] **T8.3**: Deployment preparation
  - Package model artifacts
  - Create deployment configuration
  - Document system requirements
  - Create quick start guide
  - **Validation**: Model ready for deployment

---

## Dependencies Between Tasks

```
T0.1, T0.2 (Baseline)
    ↓
T1.1, T1.2, T1.3 (Core Model) ← Independent
    ↓
T2.1, T2 2, T2.3 (FL Integration)
    ↓
T3.1, T3.2 (Data Pipeline)
    ↓
T4.1, T4.2 (Training Infrastructure)
    ↓
T5.1, T5.2 (Evaluation)
    ↓
T6.1 (Small Dataset Test) → T6.2 (Full Dataset Test) → T6.3 (Integration Tests)
    ↓
T7.1, T7.2, T7.3 (Documentation)
    ↓
T8.1, T8.2, T8.3 (Finalization)
```

## Parallelizable Work

- T1.1, T1.2, T1.3 can be done in parallel
- T2.1, T2.2 can be done in parallel (T2.3 depends on both)
- T5.1, T5.2 can be done in parallel
- T7.1, T7.2, T7.3 can be done in parallel

## Estimated Timeline

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| Pre-Implementation | T0.1 - T0.2 | 1 day |
| Core TabTransformer | T1.1 - T1.3 | 2 days |
| FL Integration | T2.1 - T2.3 | 1.5 days |
| Data Pipeline | T3.1 - T3.2 | 1 day |
| Training Infrastructure | T4.1 - T4.2 | 1 day |
| Evaluation | T5.1 - T5.2 | 0.5 days |
| Testing & Validation | T6.1 - T6.3 | 2 days |
| Documentation | T7.1 - T7.3 | 1 day |
| Cleanup & Finalization | T8.1 - T8.3 | 0.5 days |
| **Total** | | **~10.5 days** |

## Critical Path

T0.1 → T1.1 → T2.1 → T3.1 → T4.1 → T6.1 → T6.2 → T8.2

Minimum time if everything goes smoothly: **7 days**

## Test Commands

```bash
# Unit tests
pytest Notebooks/tests/test_tabtransformer.py -v
pytest Notebooks/tests/test_fl_pytorch.py -v

# Integration test (small dataset)
jupyter nbconvert --to notebook --execute Notebooks/2_Federated_Training.ipynb

# Full validation (after code complete)
./scripts/run_all_tests.sh
```
