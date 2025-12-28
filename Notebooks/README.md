# Federated Learning Training Infrastructure

## 📋 Overview

This directory contains the complete implementation for training a Federated Learning model to detect IoT network attacks using the CICIoT2023 dataset.

## 🗂️ Directory Structure

```
Notebooks/
├── 1_Data_Preprocessing.ipynb       # [TODO] Data loading, cleaning, partitioning
├── 2_Federated_Training.ipynb       # [TODO] FL training loop
├── 3_Model_Evaluation_Export.ipynb  # [TODO] Evaluation and export
├── includes.py                       # ✅ Constants and attack class mappings
├── utils/                            # ✅ Utility modules
│   ├── __init__.py                  # Package initialization
│   ├── data_utils.py                # Data processing functions
│   ├── fl_utils.py                  # FL client/server logic
│   └── model_utils.py               # Model architecture
└── configs/                          # ✅ Configuration files
    └── training_config.yaml         # Hyperparameters and settings
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- 16GB+ RAM (recommended)
- GPU (optional, but recommended for faster training)
- Dataset: CICIoT2023 (~12GB, 169 CSV files in `../DataSets/`)

### Installation

```bash
# Install required packages
pip3 install tensorflow keras scikit-learn pandas numpy matplotlib seaborn pyyaml

# Verify installation
python3 -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed')"
```

### Usage

Run the notebooks in order:

#### 1️⃣ Data Preprocessing
```bash
jupyter notebook 1_Data_Preprocessing.ipynb
```

**What it does:**
- Loads 169 CSV files using chunking (memory-efficient)
- Cleans data (removes nulls, duplicates)
- Encodes 34 attack classes to numeric labels
- Normalizes features using MinMaxScaler
- Partitions data for 5 clients (Non-IID distribution)
- Saves preprocessed data to `../Output/data/`

**Outputs:**
- `../Output/data/client_0_data.npz` ... `client_4_data.npz`
- `../Output/data/test_data.npz`
- `../Output/models/scaler.pkl`
- `../Output/models/label_encoder.pkl`
- `../Output/models/labels.json`

#### 2️⃣ Federated Training
```bash
jupyter notebook 2_Federated_Training.ipynb
```

**What it does:**
- Loads training configuration from `configs/training_config.yaml`
- Initializes FederatedServer with global model
- Creates 5 FederatedClient instances
- Runs FL training loop (30-50 rounds)
  - Server broadcasts model → Clients train locally → Server aggregates (FedAvg)
- Evaluates on test set after each round
- Saves final model and training history

**Outputs:**
- `../Output/models/global_model.h5`
- `../Output/metrics/training_history.json`

**Expected training time:** 4-6 hours (GPU) | 8-12 hours (CPU)

#### 3️⃣ Model Evaluation & Export
```bash
jupyter notebook 3_Model_Evaluation_Export.ipynb
```

**What it does:**
- Loads trained model
- Generates predictions on test set
- Calculates metrics (Accuracy, Precision, Recall, F1-Score)
- Creates visualizations:
  - Confusion matrix (34x34)
  - Training curves
  - Per-class F1-Score bar chart
- Exports all metrics and plots

**Outputs:**
- `../Output/metrics/confusion_matrix.png`
- `../Output/metrics/accuracy_plot.png`
- `../Output/metrics/f1_scores_per_class.png`
- `../Output/metrics/metrics_report.json`

## ⚙️ Configuration

Edit `configs/training_config.yaml` to customize:

```yaml
# Key parameters
num_clients: 5          # Number of simulated clients
num_rounds: 30          # FL communication rounds
local_epochs: 5         # Training epochs per client per round
batch_size: 256         # Batch size
learning_rate: 0.001    # Learning rate

# Model architecture
model:
  hidden_layers: [128, 64, 32]  # Hidden layer sizes
  dropout_rate: 0.3             # Dropout rate
```

## 📊 Expected Results

| Metric | Target | Notes |
|--------|--------|-------|
| Overall Accuracy | >95% | On test set |
| Per-class F1-Score | >0.85 | For all 34 classes |
| Training Time | <6 hours | With GPU |
| Model Size | ~10-20 MB | .h5 file |

## 🧪 Testing Strategy

### Phase 1: Quick Test (10% data)
```yaml
# In training_config.yaml
experimental:
  use_sample_data: true
  sample_fraction: 0.1
```
- Training time: ~30 mins
- Expected accuracy: ~85-90%
- Purpose: Verify pipeline works

### Phase 2: Validation (50% data)
```yaml
experimental:
  use_sample_data: true
  sample_fraction: 0.5
```
- Training time: ~2-3 hours
- Expected accuracy: ~92-93%
- Purpose: Verify convergence

### Phase 3: Production (100% data)
```yaml
experimental:
  use_sample_data: false
```
- Training time: ~5-6 hours
- Expected accuracy: >95%
- Purpose: Final model for deployment

## 🐛 Troubleshooting

### Out of Memory
```python
# Reduce chunk size in data_utils.py
chunk_size = 10000  # Default: 50000

# Or use sample data
use_sample_data = true
sample_fraction = 0.1
```

### Training Not Converging
```yaml
# Increase rounds
num_rounds: 50  # Default: 30

# Or reduce learning rate
learning_rate: 0.0005  # Default: 0.001
```

### Low F1 for Minority Classes
```python
# In model training, use class weights
class_weight = 'balanced'
```

## 📚 Module Documentation

### data_utils.py
- `load_dataset_chunked()` - Load CSV files with chunking
- `clean_data()` - Remove nulls and duplicates
- `encode_labels()` - Map attack names to numeric
- `normalize_features()` - MinMaxScaler normalization
- `partition_data_noniid()` - Non-IID data partitioning
- `save_partitioned_data()` - Save as .npz files

### model_utils.py
- `create_dnn_model()` - Build DNN architecture
- `compile_model()` - Configure optimizer and loss
- `get_model_weights()` - Extract weights
- `set_model_weights()` - Set weights
- `save_model()` / `load_model()` - Model I/O

### fl_utils.py
- `FederatedClient` - Client-side training
- `FederatedServer` - Server-side aggregation (FedAvg)
- `federated_training_loop()` - Main FL orchestrator

## 🎯 Success Criteria

- [ ] All 3 notebooks run without errors
- [ ] Model achieves >95% accuracy on test set
- [ ] All 34 classes have F1-Score >0.85
- [ ] All output files generated in `../Output/`
- [ ] Confusion matrix shows good class separation
- [ ] Exported model loadable and usable for inference

## 📝 Notes

- **Hardware**: 16GB+ RAM recommended, GPU optional
- **Dataset**: Must have 169 CSV files in `../DataSets/`
- **Time**: Allow 1-2 weeks for full implementation and testing
- **Checkpoints**: Model saved every 10 rounds (configurable)

## 🔗 References

- FedAvg Paper: McMahan et al. (2017)
- CICIoT2023 Dataset: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- OpenSpec Proposal: `../openspec/changes/build-fl-training-infrastructure/`

## 👤 Author

**Nguyen Duc Thang**  
Project: IoT Network Attack Detection using Federated Learning  
Date: December 2025

---

**Status:** ✅ Infrastructure setup complete | 📝 Notebooks in progress
