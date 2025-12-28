# Change: Build Complete Federated Learning Training Infrastructure

## Why

Hiện tại project đang có các notebook rời rạc (`01-Data_Exploration.ipynb`, `03-Federated_Learning_UPDATED.ipynb`, v.v.) nhưng chưa có một hệ thống hoàn chỉnh để training model Federated Learning một cách có tổ chức. Cần xây dựng một infrastructure đầy đủ để:

1. Chuẩn bị và xử lý dataset CICIoT2023 (~12GB CSV files)
2. Huấn luyện model FL với kiến trúc rõ ràng (server-client simulation)
3. Đánh giá model với các metrics chi tiết
4. Xuất ra các artifacts cần thiết cho việc tích hợp vào Web App sau này

Target: Đạt độ chính xác > 95% trên test set với khả năng phát hiện 34 loại tấn công IoT.

## What Changes

### Cấu trúc thư mục mới
```
Notebooks/
├── 1_Data_Preprocessing.ipynb       # Data loading, cleaning, normalization, partitioning
├── 2_Federated_Training.ipynb       # FL training loop (FedAvg algorithm)
├── 3_Model_Evaluation_Export.ipynb  # Evaluation, visualization, export
├── includes.py                       # [Existing] Constants and mappings
├── utils/
│   ├── data_utils.py                # Data loading and preprocessing functions
│   ├── fl_utils.py                  # FL client/server logic
│   └── model_utils.py               # Model architecture definitions
└── configs/
    └── training_config.yaml         # Hyperparameters and config

Output/
├── models/
│   ├── global_model.h5              # Final trained model
│   ├── scaler.pkl                   # MinMaxScaler for normalization
│   └── label_encoder.pkl            # Label encoder
├── metrics/
│   ├── training_history.json        # Loss/accuracy per round
│   ├── confusion_matrix.png         # Confusion matrix visualization
│   ├── metrics_report.json          # Precision, Recall, F1-Score
│   └── accuracy_plot.png            # Training curves
└── data/
    ├── client_1_data.npz            # Partitioned client data
    ├── client_2_data.npz
    ├── client_3_data.npz
    ├── client_4_data.npz
    ├── client_5_data.npz
    └── test_data.npz                # Test set
```

### Components chính

#### 1. Data Preprocessing Pipeline
- **Load**: Đọc 169 CSV files từ `DataSets/` (sử dụng pandas chunking để tránh OOM)
- **Clean**: Xử lý missing values, loại bỏ duplicate rows
- **Feature Selection**: Sử dụng 46 features từ `includes.py::X_columns`
- **Label Encoding**: Map 34 attack classes → numeric labels (0-33)
- **Normalization**: MinMaxScaler cho tất cả features
- **Data Partitioning**: Chia dữ liệu cho 5 clients (Non-IID distribution để giả lập thực tế)
- **Train/Test Split**: 80% train, 20% test

#### 2. Federated Learning Training
- **Model Architecture**: Deep Neural Network
  - Input Layer: 46 features
  - Hidden Layers: [128, 64, 32] units với Dropout (0.3)
  - Output Layer: 34 units (Softmax)
- **FL Algorithm**: FedAvg (Federated Averaging)
  - Number of clients: 5
  - Number of rounds: 30-50
  - Local epochs per round: 5
  - Batch size: 256
  - Learning rate: 0.001 (Adam optimizer)
- **Client Logic**: 
  - Load global model
  - Train on local data
  - Compute local weights
  - Send updates to server
- **Server Logic**:
  - Aggregate client weights (average)
  - Update global model
  - Broadcast to clients

#### 3. Model Evaluation & Export
- **Metrics**:
  - Overall Accuracy
  - Per-class Precision, Recall, F1-Score
  - Confusion Matrix (34x34)
  - Classification Report
- **Visualizations**:
  - Training curves (Loss & Accuracy vs Rounds)
  - Confusion Matrix heatmap
  - Per-class F1-Score bar chart
- **Exports**:
  - `global_model.h5`: Keras model
  - `scaler.pkl`: Fitted scaler
  - `label_encoder.pkl`: Label mappings
  - `metrics_report.json`: All metrics
  - Visualization images

### Technology Stack
- **Core ML**: TensorFlow/Keras, scikit-learn, numpy, pandas
- **Visualization**: matplotlib, seaborn
- **Data**: PyYAML (config), pickle (serialization)
- **Environment**: Jupyter Notebook, Python 3.8+

## Impact

### Affected specs
- **NEW**: `specs/model-training/spec.md` - Defines requirements for FL training pipeline

### Affected code
- **NEW**: 
  - `Notebooks/1_Data_Preprocessing.ipynb`
  - `Notebooks/2_Federated_Training.ipynb`
  - `Notebooks/3_Model_Evaluation_Export.ipynb`
  - `Notebooks/utils/data_utils.py`
  - `Notebooks/utils/fl_utils.py`
  - `Notebooks/utils/model_utils.py`
  - `Notebooks/configs/training_config.yaml`
- **EXISTING**: `Notebooks/includes.py` (no changes, just referenced)

### Dependencies
- Dataset must be present in `DataSets/` (169 CSV files, ~12GB)
- Python packages: tensorflow>=2.10, keras, scikit-learn, pandas, numpy, matplotlib, seaborn, pyyaml

### Success Criteria
- ✅ Model achieves >95% accuracy on test set
- ✅ All 34 attack classes have F1-Score >0.85
- ✅ Training completes in <6 hours on GPU
- ✅ All required output files generated in `Output/` directory
- ✅ Confusion matrix shows minimal misclassification between Normal and Attack traffic
- ✅ Exported model can be loaded and used for inference

## Notes

> [!IMPORTANT]
> Dataset size: ~12GB CSV files. Cần đảm bảo đủ RAM (recommend 16GB+) hoặc sử dụng chunking strategy khi load data.

> [!WARNING]
> Non-IID data partitioning có thể ảnh hưởng đến convergence. Cần monitor training curves carefully và adjust number of rounds nếu cần.

> [!TIP]
> Sau khi train xong, nhớ lưu training logs và metrics để viết báo cáo. Files trong `Output/metrics/` sẽ được dùng để demo trong báo cáo đồ án.
