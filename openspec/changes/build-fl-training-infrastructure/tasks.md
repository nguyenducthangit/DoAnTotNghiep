# Implementation Tasks

## 1. Setup & Configuration
- [ ] 1.1 Create `Notebooks/utils/` directory structure
- [ ] 1.2 Create `Notebooks/configs/` directory structure
- [ ] 1.3 Create `training_config.yaml` with all hyperparameters
- [ ] 1.4 Verify dataset files in `DataSets/` (169 CSV files)
- [ ] 1.5 Verify Python dependencies installed (tensorflow, keras, scikit-learn, etc.)

## 2. Data Preprocessing Module
- [ ] 2.1 Create `Notebooks/utils/data_utils.py`
  - [ ] 2.1.1 Implement `load_dataset_chunked()` - Load 169 CSV files with memory efficiency
  - [ ] 2.1.2 Implement `clean_data()` - Handle missing values and duplicates
  - [ ] 2.1.3 Implement `encode_labels()` - Map attack names to numeric (0-33)
  - [ ] 2.1.4 Implement `normalize_features()` - MinMaxScaler fitting and transform
  - [ ] 2.1.5 Implement `partition_data_noniid()` - Split data for 5 clients (Non-IID)
  - [ ] 2.1.6 Implement `save_partitioned_data()` - Save as .npz files
- [ ] 2.2 Create `Notebooks/1_Data_Preprocessing.ipynb`
  - [ ] 2.2.1 Load raw CSV files using chunking
  - [ ] 2.2.2 Perform data cleaning and exploration (check class distribution)
  - [ ] 2.2.3 Apply label encoding and save encoder
  - [ ] 2.2.4 Apply normalization and save scaler
  - [ ] 2.2.5 Partition data into 5 clients + test set
  - [ ] 2.2.6 Save all partitioned data to `Output/data/`
  - [ ] 2.2.7 Generate data statistics report (class distribution per client)

## 3. Model Architecture
- [ ] 3.1 Create `Notebooks/utils/model_utils.py`
  - [ ] 3.1.1 Implement `create_dnn_model()` - Build DNN architecture
    - Input: 46 features
    - Hidden layers: [128, 64, 32] with ReLU + Dropout(0.3)
    - Output: 34 units with Softmax
  - [ ] 3.1.2 Implement `compile_model()` - Configure optimizer, loss, metrics
  - [ ] 3.1.3 Implement `get_model_weights()` - Extract model weights
  - [ ] 3.1.4 Implement `set_model_weights()` - Set model weights

## 4. Federated Learning Logic
- [ ] 4.1 Create `Notebooks/utils/fl_utils.py`
  - [ ] 4.1.1 Implement `FederatedClient` class
    - `__init__()` - Initialize with local data
    - `local_train()` - Train on local data for E epochs
    - `get_weights()` - Return local model weights
  - [ ] 4.1.2 Implement `FederatedServer` class
    - `__init__()` - Initialize global model
    - `aggregate_weights()` - FedAvg: Average client weights
    - `update_global_model()` - Update global model with aggregated weights
    - `evaluate_global_model()` - Test on test set
  - [ ] 4.1.3 Implement `federated_training_loop()` - Main FL training orchestrator
- [ ] 4.2 Create `Notebooks/2_Federated_Training.ipynb`
  - [ ] 4.2.1 Load training config from YAML
  - [ ] 4.2.2 Load partitioned client data
  - [ ] 4.2.3 Initialize FederatedServer with global model
  - [ ] 4.2.4 Initialize 5 FederatedClient instances
  - [ ] 4.2.5 Run federated training loop (30-50 rounds)
    - Server broadcasts global model
    - Each client trains locally
    - Clients send weights to server
    - Server aggregates and updates global model
    - Evaluate on test set each round
  - [ ] 4.2.6 Save training history (loss, accuracy per round)
  - [ ] 4.2.7 Save final global model to `Output/models/global_model.h5`

## 5. Model Evaluation & Visualization
- [ ] 5.1 Create `Notebooks/3_Model_Evaluation_Export.ipynb`
  - [ ] 5.1.1 Load trained global model
  - [ ] 5.1.2 Load test set
  - [ ] 5.1.3 Generate predictions
  - [ ] 5.1.4 Calculate metrics:
    - Overall accuracy
    - Per-class Precision, Recall, F1-Score
    - Classification report
  - [ ] 5.1.5 Create confusion matrix (34x34)
  - [ ] 5.1.6 Visualize confusion matrix heatmap
  - [ ] 5.1.7 Plot training curves (Accuracy & Loss vs Rounds)
  - [ ] 5.1.8 Plot per-class F1-Score bar chart
  - [ ] 5.1.9 Save all metrics to `Output/metrics/metrics_report.json`
  - [ ] 5.1.10 Save all visualizations to `Output/metrics/`

## 6. Export Artifacts
- [ ] 6.1 Verify `Output/models/global_model.h5` exists and loadable
- [ ] 6.2 Verify `Output/models/scaler.pkl` exists and loadable
- [ ] 6.3 Verify `Output/models/label_encoder.pkl` exists and loadable
- [ ] 6.4 Create `Output/models/labels.json` - Human-readable label mapping
  - Format: `{"0": "BenignTraffic", "1": "DDoS-RSTFINFlood", ...}`
- [ ] 6.5 Verify all metric files in `Output/metrics/` directory

## 7. Testing & Validation
- [ ] 7.1 Test model loading and inference on sample data
- [ ] 7.2 Verify model achieves >95% accuracy on test set
- [ ] 7.3 Verify all 34 classes have F1-Score >0.85
- [ ] 7.4 Verify confusion matrix shows good separation
- [ ] 7.5 Verify exported files can be used independently
- [ ] 7.6 Document any classes with low performance (<0.85 F1)

## 8. Documentation
- [ ] 8.1 Add README.md in `Notebooks/` explaining:
  - How to run each notebook in order
  - Hardware requirements (RAM, GPU)
  - Expected training time
  - Output file descriptions
- [ ] 8.2 Add inline comments in all Python utility files
- [ ] 8.3 Add markdown cells in notebooks explaining each step
- [ ] 8.4 Create summary report for thesis:
  - Final accuracy
  - Training time
  - Key visualizations
  - Lessons learned

## Dependencies
- Sequential: Tasks must be done in order (1 → 2 → 3 → 4 → 5 → 6 → 7 → 8)
- Critical path: Data preprocessing → Model architecture → FL training → Evaluation
- Parallelizable: Visualization tasks (5.1.6, 5.1.7, 5.1.8) can be done in parallel once predictions are ready
