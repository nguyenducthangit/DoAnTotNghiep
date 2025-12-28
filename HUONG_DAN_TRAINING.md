# HƯỚNG DẪN TRAINING MODEL FEDERATED LEARNING
## Phát Hiện Tấn Công Mạng IoT

**Tác giả:** Nguyễn Đức Thắng  
**Ngày:** 28/12/2025  
**Phiên bản:** 1.0

---

## 📋 MỤC LỤC

1. [Giới Thiệu](#1-giới-thiệu)
2. [Yêu Cầu Hệ Thống](#2-yêu-cầu-hệ-thống)
3. [Cài Đặt Môi Trường](#3-cài-đặt-môi-trường)
4. [Cấu Trúc Thư Mục](#4-cấu-trúc-thư-mục)
5. [Cấu Hình Training](#5-cấu-hình-training)
6. [Hướng Dẫn Chạy Từng Bước](#6-hướng-dẫn-chạy-từng-bước)
7. [Xử Lý Lỗi Thường Gặp](#7-xử-lý-lỗi-thường-gặp)
8. [Kiểm Tra Kết Quả](#8-kiểm-tra-kết-quả)
9. [Tips & Best Practices](#9-tips--best-practices)

---

## 1. GIỚI THIỆU

### 1.1 Mục Đích
Hướng dẫn này giúp bạn training model Federated Learning để phát hiện 34 loại tấn công mạng IoT sử dụng dataset CICIoT2023.

### 1.2 Quy Trình Tổng Quan
```
Bước 1: Data Preprocessing (30-60 phút)
   ↓
Bước 2: Federated Training (4-6 giờ)
   ↓
Bước 3: Model Evaluation (10-20 phút)
   ↓
Kết quả: Model + Metrics + Visualizations
```

### 1.3 Kết Quả Mong Đợi
- ✅ Model accuracy > 95%
- ✅ F1-Score > 0.85 cho tất cả 34 classes
- ✅ Files xuất ra cho Web App integration

---

## 2. YÊU CẦU HỆ THỐNG

### 2.1 Phần Cứng (Tối Thiểu)
- **CPU:** Intel Core i5 hoặc tương đương
- **RAM:** 16GB (khuyến nghị 32GB)
- **Ổ cứng:** 20GB trống
- **GPU:** Không bắt buộc (nhưng khuyến nghị để training nhanh hơn)

### 2.2 Phần Cứng (Khuyến Nghị)
- **CPU:** Intel Core i7/i9 hoặc AMD Ryzen 7/9
- **RAM:** 32GB+
- **GPU:** NVIDIA GPU với CUDA support (GTX 1060 trở lên)
- **Ổ cứng:** SSD với 50GB trống

### 2.3 Phần Mềm
- **OS:** macOS, Linux, hoặc Windows 10/11
- **Python:** 3.8, 3.9, hoặc 3.10
- **Jupyter Notebook:** Phiên bản mới nhất
- **Git:** Để quản lý code (optional)

### 2.4 Dataset
- **Tên:** CICIoT2023
- **Kích thước:** ~12GB (169 CSV files)
- **Vị trí:** Phải có trong thư mục `DataSets/`

---

## 3. CÀI ĐẶT MÔI TRƯỜNG

### 3.1 Kiểm Tra Python Version

Mở Terminal và chạy:

```bash
python3 --version
```

**Kết quả mong đợi:** Python 3.8.x, 3.9.x, hoặc 3.10.x

⚠️ **Nếu không có Python 3:** Download từ https://www.python.org/downloads/

---

### 3.2 Cài Đặt Dependencies

#### Bước 1: Mở Terminal
```bash
cd "/Users/user/Documents/ĐỒ ÁN/Do an"
```

#### Bước 2: Cài đặt packages
```bash
pip3 install tensorflow keras scikit-learn pandas numpy matplotlib seaborn pyyaml jupyter
```

**Thời gian:** 5-10 phút

#### Bước 3: Verify cài đặt
```bash
python3 -c "import tensorflow as tf; import keras; import sklearn; import pandas; import numpy; import matplotlib; import seaborn; import yaml; print('✅ All packages installed successfully!')"
```

**Kết quả mong đợi:**
```
✅ All packages installed successfully!
```

---

### 3.3 Kiểm Tra GPU (Optional nhưng khuyến nghị)

```bash
python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPU available: {len(gpus) > 0}'); print(f'Number of GPUs: {len(gpus)}')"
```

**Nếu có GPU:**
```
GPU available: True
Number of GPUs: 1
```

**Nếu không có GPU:**
```
GPU available: False
Number of GPUs: 0
```

⚠️ **Lưu ý:** Không có GPU vẫn chạy được, nhưng sẽ chậm hơn 3-4 lần.

---

### 3.4 Verify Dataset

```bash
ls -lh DataSets/*.csv | wc -l
```

**Kết quả mong đợi:** `169`

⚠️ **Nếu không đủ 169 files:** Kiểm tra lại dataset download.

---

## 4. CẤU TRÚC THƯ MỤC

Sau khi setup xong, cấu trúc thư mục của bạn sẽ như sau:

```
Do an/
├── DataSets/                          # Dataset (169 CSV files)
│   ├── part-00000-*.csv
│   ├── part-00001-*.csv
│   └── ... (169 files total)
│
├── Notebooks/                         # Code và notebooks
│   ├── 1_Data_Preprocessing.ipynb    # ⭐ Notebook 1
│   ├── 2_Federated_Training.ipynb    # ⭐ Notebook 2
│   ├── 3_Model_Evaluation_Export.ipynb # ⭐ Notebook 3
│   ├── includes.py                    # Constants
│   ├── README.md                      # Documentation
│   ├── configs/
│   │   └── training_config.yaml      # ⚙️ Configuration
│   └── utils/
│       ├── __init__.py
│       ├── data_utils.py             # Data processing
│       ├── model_utils.py            # Model architecture
│       └── fl_utils.py               # FL logic
│
└── Output/                            # Kết quả training
    ├── models/                        # Model files
    ├── metrics/                       # Metrics và plots
    └── data/                          # Partitioned data
```

---

## 5. CẤU HÌNH TRAINING

### 5.1 File Cấu Hình

File cấu hình chính: `Notebooks/configs/training_config.yaml`

### 5.2 Các Tham Số Quan Trọng

#### **Federated Learning Settings**
```yaml
num_clients: 5          # Số lượng clients (devices) giả lập
num_rounds: 30          # Số vòng training (tăng lên 50 nếu cần)
local_epochs: 5         # Số epochs mỗi client train
batch_size: 256         # Batch size
```

#### **Model Architecture**
```yaml
model:
  input_dim: 46              # Số features
  hidden_layers: [128, 64, 32]  # Kích thước hidden layers
  num_classes: 34            # Số loại tấn công
  dropout_rate: 0.3          # Dropout để tránh overfitting
```

#### **Optimizer**
```yaml
optimizer:
  type: adam
  learning_rate: 0.001      # Learning rate
```

#### **Data Processing**
```yaml
data:
  test_split_ratio: 0.2     # 20% data cho test
  chunk_size: 50000         # Chunk size khi load CSV
  partition_strategy: non_iid  # Non-IID distribution
```

---

### 5.3 Chế Độ Test (Khuyến Nghị Cho Lần Đầu)

**⚠️ QUAN TRỌNG:** Lần đầu chạy, nên test với 10% data trước!

Mở file `Notebooks/configs/training_config.yaml` và sửa:

```yaml
experimental:
  use_sample_data: true     # ← Đổi thành true
  sample_fraction: 0.1      # Dùng 10% data
```

**Lợi ích:**
- Training chỉ mất ~30-60 phút (thay vì 5-6 giờ)
- Verify pipeline hoạt động đúng
- Phát hiện lỗi sớm

**Sau khi test xong, đổi lại:**
```yaml
experimental:
  use_sample_data: false    # ← Đổi thành false để train full
```

---

## 6. HƯỚNG DẪN CHẠY TỪNG BƯỚC

### 📌 BƯỚC 1: DATA PREPROCESSING

**Mục đích:** Load, clean, và partition data cho FL training

#### 1.1 Mở Jupyter Notebook

```bash
cd "/Users/user/Documents/ĐỒ ÁN/Do an/Notebooks"
jupyter notebook
```

**Kết quả:** Browser sẽ mở với Jupyter interface.

---

#### 1.2 Mở Notebook 1

Trong Jupyter, click vào: `1_Data_Preprocessing.ipynb`

---

#### 1.3 Chạy Từng Cell

**⚠️ QUAN TRỌNG:** Chạy từng cell theo thứ tự từ trên xuống dưới!

**Cách chạy:**
- Click vào cell
- Nhấn `Shift + Enter` (hoặc click nút ▶️ Run)
- Đợi cell chạy xong (dấu `*` biến thành số)
- Chuyển sang cell tiếp theo

---

#### 1.4 Các Cell Quan Trọng

**Cell 1-2: Setup and Imports**
```
✅ All imports successful!
   Number of features: 46
   Label column: label
   Number of attack classes: 34
```

**Cell 3: Load Dataset**
```
📂 Loading dataset from: ../DataSets
   Found 169 CSV files
   [1/169] Loading part-00000-*.csv... ✓ 123,456 rows
   ...
   Total rows: 12,345,678
```

⏱️ **Thời gian:** 10-30 phút (tùy RAM và CPU)

**Cell 5: Encode Labels**
```
🏷️  Encoding labels...
   Found 34 unique labels:
   ✓ Encoded 34 classes to numeric values (0-33)
   💾 Saved label encoder to: ../Output/models/label_encoder.pkl
   💾 Saved label mapping to: ../Output/models/labels.json
```

**Cell 6: Normalize Features**
```
📏 Normalizing features...
   ✓ Normalized 46 features to [0, 1] range
   💾 Saved scaler to: ../Output/models/scaler.pkl
```

**Cell 7: Partition Data**
```
🔀 Partitioning data for 5 clients (Non-IID)...
   Train set: 9,876,543 samples
   Test set: 2,469,135 samples
   Client 0 (DDoS): 1,234,567 samples
   Client 1 (Recon): 1,123,456 samples
   ...
```

**Cell 8: Save Data**
```
💾 Saving partitioned data to: ../Output/data
   ✓ Saved client_0: ../Output/data/client_0_data.npz (123,456 samples)
   ...
   ✅ All data saved successfully!
```

---

#### 1.5 Verify Outputs

**Cell cuối (Verification):**
```
🔍 Verifying saved files...

📂 Data files:
   ✓ client_0_data.npz (234.56 MB)
   ✓ client_1_data.npz (223.45 MB)
   ✓ client_2_data.npz (245.67 MB)
   ✓ client_3_data.npz (212.34 MB)
   ✓ client_4_data.npz (256.78 MB)
   ✓ test_data.npz (567.89 MB)

📂 Model artifacts:
   ✓ scaler.pkl (12.34 KB)
   ✓ label_encoder.pkl (5.67 KB)
   ✓ labels.json (1.23 KB)

✅ Verification complete!
```

---

#### 1.6 Kiểm Tra Thư Mục Output

```bash
ls -lh ../Output/data/
ls -lh ../Output/models/
```

**Phải có:**
- 6 files `.npz` trong `Output/data/`
- 3 files (scaler, encoder, labels) trong `Output/models/`

---

### ✅ CHECKPOINT 1: Data Preprocessing Hoàn Thành

Nếu tất cả cells chạy thành công và có đủ files → Chuyển sang Bước 2!

---

### 📌 BƯỚC 2: FEDERATED LEARNING TRAINING

**Mục đích:** Train model sử dụng Federated Learning (FedAvg)

⏱️ **Thời gian dự kiến:**
- Với 10% data: 30-60 phút
- Với 100% data + GPU: 4-6 giờ
- Với 100% data + CPU: 8-12 giờ

---

#### 2.1 Mở Notebook 2

Trong Jupyter, click vào: `2_Federated_Training.ipynb`

---

#### 2.2 Chạy Từng Cell

**Cell 1: Setup and Imports**
```
✅ GPU available: 1 device(s)  # Hoặc "No GPU found" nếu dùng CPU
✅ TensorFlow version: 2.x.x
✅ Keras version: 2.x.x
```

**Cell 2: Load Configuration**
```
📄 Configuration loaded:

🔧 FL Settings:
   Number of clients: 5
   Number of rounds: 30
   Local epochs: 5
   Batch size: 256

🏗️  Model Architecture:
   Input dim: 46
   Hidden layers: [128, 64, 32]
   Output classes: 34
   Dropout rate: 0.3
```

**Cell 3: Load Preprocessed Data**
```
📂 Loading client data...

   ✓ client_0: 1,234,567 samples
   ✓ client_1: 1,123,456 samples
   ✓ client_2: 1,345,678 samples
   ✓ client_3: 1,234,567 samples
   ✓ client_4: 1,456,789 samples
   ✓ test: 2,469,135 samples

✅ All data loaded successfully!
```

**Cell 4: Create Global Model**
```
🏗️  Creating DNN model...
   Input dimension: 46
   Hidden layers: [128, 64, 32]
   Output classes: 34
   Dropout rate: 0.3
   ✓ Model created with 3 hidden layers

⚙️  Compiling model...
   Optimizer: Adam (lr=0.001)
   Loss: sparse_categorical_crossentropy
   Metrics: ['accuracy']
   ✓ Model compiled successfully

MODEL ARCHITECTURE SUMMARY
================================================================
Layer (type)                 Output Shape              Param #   
================================================================
dense_1 (Dense)              (None, 128)               6016      
dropout_1 (Dropout)          (None, 128)               0         
dense_2 (Dense)              (None, 64)                8256      
dropout_2 (Dropout)          (None, 64)                0         
dense_3 (Dense)              (None, 32)                2080      
dropout_3 (Dropout)          (None, 32)                0         
output (Dense)               (None, 34)                1122      
================================================================
Total parameters: 17,474
Estimated model size: 0.07 MB
```

**Cell 5: Initialize Server and Clients**
```
🖥️  Initializing Federated Server...
   ✓ Server initialized

👥 Initializing Federated Clients...
   Client 0 initialized with 1,234,567 samples
   Client 1 initialized with 1,123,456 samples
   ...

✅ 5 clients initialized!
```

---

#### 2.3 Cell 6: Main Training Loop ⚠️ QUAN TRỌNG

**Đây là cell mất nhiều thời gian nhất!**

```
🕐 Training started at: 2025-12-28 01:30:00

================================================================================
FEDERATED LEARNING TRAINING
================================================================================
Number of clients: 5
Number of rounds: 30
Local epochs per round: 5
Batch size: 256
Test set size: 2,469,135
================================================================================

================================================================================
ROUND 1/30
================================================================================
📡 Broadcasting global model to 5 clients...
   ✓ Server initialized

   Client 0 training... ✓ Loss: 2.3456, Acc: 0.3456
   Client 1 training... ✓ Loss: 2.4567, Acc: 0.3234
   Client 2 training... ✓ Loss: 2.3789, Acc: 0.3567
   Client 3 training... ✓ Loss: 2.4012, Acc: 0.3345
   Client 4 training... ✓ Loss: 2.3890, Acc: 0.3478

🔄 Aggregating weights from 5 clients...
   ✓ Global model updated

📊 Evaluating global model on test set...

────────────────────────────────────────────────────────────────────────────────
ROUND 1 SUMMARY:
   Global Test Loss: 2.3945
   Global Test Accuracy: 0.3456 (34.56%)
   Avg Client Loss: 2.3943
   Avg Client Accuracy: 0.3416
────────────────────────────────────────────────────────────────────────────────

... (Rounds 2-29 tương tự)

================================================================================
ROUND 30/30
================================================================================
...
────────────────────────────────────────────────────────────────────────────────
ROUND 30 SUMMARY:
   Global Test Loss: 0.1234
   Global Test Accuracy: 0.9678 (96.78%)  ← Mục tiêu >95%!
   Avg Client Loss: 0.1245
   Avg Client Accuracy: 0.9654
────────────────────────────────────────────────────────────────────────────────

================================================================================
TRAINING COMPLETED!
================================================================================
Final Test Accuracy: 96.78%
================================================================================

🕐 Training completed at: 2025-12-28 07:30:00
⏱️  Total training time: 6:00:00
   (360.00 minutes)
```

**⏱️ Theo dõi tiến độ:**
- Mỗi round mất ~10-15 phút (với full data + GPU)
- Accuracy sẽ tăng dần qua các rounds
- Nếu accuracy không tăng sau 10 rounds → Có vấn đề, xem phần Troubleshooting

---

#### 2.4 Các Cell Tiếp Theo

**Cell 7: Visualize Training Progress**

Sẽ hiển thị 2 biểu đồ:
- Accuracy vs Round (đường tăng dần, vượt 95% line)
- Loss vs Round (đường giảm dần)

**Cell 8: Save Trained Model**
```
💾 Global model saved to: ../Output/models/global_model.h5

✅ Model saved successfully!
   Path: ../Output/models/global_model.h5
   Size: 12.34 MB
```

**Cell 9: Save Training History**
```
💾 Training history saved to: ../Output/metrics/training_history.json
```

**Cell 10: Quick Evaluation**
```
🔍 Loading saved model for verification...
📂 Loading model from: ../Output/models/global_model.h5
   ✓ Model loaded successfully

📊 Evaluating on test set...

✅ Test Set Results:
   Loss: 0.1234
   Accuracy: 0.9678 (96.78%)

🔮 Sample predictions (first 10 test samples):
   ✓ Sample 1: Predicted=0, True=0
   ✓ Sample 2: Predicted=5, True=5
   ✗ Sample 3: Predicted=12, True=11  ← Sai 1 sample
   ...
```

---

### ✅ CHECKPOINT 2: Training Hoàn Thành

Nếu:
- ✅ Training chạy hết 30 rounds
- ✅ Final accuracy > 95%
- ✅ Model saved thành công

→ Chuyển sang Bước 3!

---

### 📌 BƯỚC 3: MODEL EVALUATION & EXPORT

**Mục đích:** Đánh giá chi tiết model và tạo visualizations cho báo cáo

⏱️ **Thời gian:** 10-20 phút

---

#### 3.1 Mở Notebook 3

Trong Jupyter, click vào: `3_Model_Evaluation_Export.ipynb`

---

#### 3.2 Chạy Từng Cell

**Cell 1-3: Setup, Load Model, Load Labels**
```
✅ All imports successful!
📂 Loading trained model from: ../Output/models/global_model.h5
   ✓ Model loaded successfully

✅ Data loaded:
   Test samples: 2,469,135
   Features: 46
   Classes: 34
```

**Cell 4: Generate Predictions**
```
🔮 Generating predictions on test set...
2469135/2469135 [==============================] - 45s 18us/sample

✅ Predictions generated!
   Prediction shape: (2469135,)
   Unique predicted classes: 34
```

**Cell 5: Calculate Overall Metrics**
```
================================================================================
OVERALL METRICS
================================================================================

📊 Overall Accuracy: 0.9678 (96.78%)  ← Đạt mục tiêu!

📈 Macro Averages (unweighted):
   Precision: 0.9534
   Recall: 0.9512
   F1-Score: 0.9523

📈 Weighted Averages (by support):
   Precision: 0.9689
   Recall: 0.9678
   F1-Score: 0.9683
================================================================================

✅ SUCCESS: Target accuracy (>95%) achieved!
```

**Cell 6: Per-Class Metrics**

Hiển thị bảng với 34 rows:
```
                    Class  Precision  Recall  F1-Score  Support
0          BenignTraffic     0.9876  0.9912    0.9894   500000
1      DDoS-RSTFINFlood     0.9654  0.9587    0.9620    75000
2      DDoS-PSHACK_Flood    0.9723  0.9698    0.9710    68000
...
33  DictionaryBruteForce    0.8912  0.8756    0.8833    12000

✅ All classes have F1-Score >= 0.85!  ← Hoặc warning nếu có class < 0.85
```

**Cell 7-8: Confusion Matrix**

Hiển thị confusion matrix 34x34 (heatmap màu xanh)

```
💾 Confusion matrix saved to: ../Output/metrics/confusion_matrix.png
```

**Cell 9: Training History Visualization**

Hiển thị 2 biểu đồ training curves

```
💾 Training curves saved to: ../Output/metrics/accuracy_plot.png
```

**Cell 10: Per-Class F1-Score Visualization**

Hiển thị bar chart với màu xanh (F1≥0.85) và đỏ (F1<0.85)

```
💾 F1-Score chart saved to: ../Output/metrics/f1_scores_per_class.png

📊 F1-Score Summary:
   Classes with F1 ≥ 0.85: 34/34 (100.0%)
   Classes with F1 < 0.85: 0/34 (0.0%)
```

**Cell 11: Export Comprehensive Metrics Report**
```
💾 Comprehensive metrics report saved to: ../Output/metrics/metrics_report.json

✅ Report includes:
   - Overall metrics (accuracy, precision, recall, F1)
   - Per-class metrics for all 34 classes
   - Confusion matrix
   - Summary statistics
```

**Cell 12: Generate Classification Report**
```
================================================================================
CLASSIFICATION REPORT
================================================================================
                       precision    recall  f1-score   support

       BenignTraffic       0.99      0.99      0.99    500000
   DDoS-RSTFINFlood       0.97      0.96      0.96     75000
  DDoS-PSHACK_Flood       0.97      0.97      0.97     68000
...
DictionaryBruteForce       0.89      0.88      0.88     12000

            accuracy                           0.97   2469135
           macro avg       0.95      0.95      0.95   2469135
        weighted avg       0.97      0.97      0.97   2469135

💾 Classification report saved to: ../Output/metrics/classification_report.txt
```

---

### ✅ CHECKPOINT 3: Evaluation Hoàn Thành

Kiểm tra thư mục `Output/`:

```bash
ls -lh ../Output/models/
ls -lh ../Output/metrics/
```

**Phải có:**

**Models:**
- ✅ `global_model.h5` (~10-20 MB)
- ✅ `scaler.pkl`
- ✅ `label_encoder.pkl`
- ✅ `labels.json`

**Metrics:**
- ✅ `training_history.json`
- ✅ `metrics_report.json`
- ✅ `classification_report.txt`
- ✅ `confusion_matrix.png`
- ✅ `accuracy_plot.png`
- ✅ `f1_scores_per_class.png`

---

## 7. XỬ LÝ LỖI THƯỜNG GẶP

### 7.1 Lỗi: Out of Memory (OOM)

**Triệu chứng:**
```
MemoryError: Unable to allocate array
```

**Nguyên nhân:** RAM không đủ để load dataset

**Giải pháp:**

**Option 1: Giảm chunk_size**
```yaml
# File: configs/training_config.yaml
data:
  chunk_size: 10000  # Giảm từ 50000 xuống 10000
```

**Option 2: Dùng sample data**
```yaml
experimental:
  use_sample_data: true
  sample_fraction: 0.1  # Chỉ dùng 10%
```

**Option 3: Close các app khác**
- Đóng browser tabs không cần thiết
- Đóng các ứng dụng nặng (Photoshop, video editors, etc.)

---

### 7.2 Lỗi: Training Không Converge

**Triệu chứng:**
- Accuracy không tăng sau 10 rounds
- Accuracy dao động không ổn định

**Giải pháp:**

**Option 1: Tăng số rounds**
```yaml
num_rounds: 50  # Tăng từ 30 lên 50
```

**Option 2: Giảm learning rate**
```yaml
optimizer:
  learning_rate: 0.0005  # Giảm từ 0.001 xuống 0.0005
```

**Option 3: Tăng local epochs**
```yaml
local_epochs: 7  # Tăng từ 5 lên 7
```

---

### 7.3 Lỗi: File Not Found

**Triệu chứng:**
```
FileNotFoundError: No such file or directory: '../DataSets'
```

**Giải pháp:**
1. Kiểm tra dataset có trong thư mục đúng không:
   ```bash
   ls -lh ../DataSets/*.csv | wc -l
   ```
2. Nếu không có, download lại dataset
3. Đảm bảo đường dẫn đúng trong notebook

---

### 7.4 Lỗi: Import Error

**Triệu chứng:**
```
ModuleNotFoundError: No module named 'tensorflow'
```

**Giải pháp:**
```bash
pip3 install tensorflow keras scikit-learn pandas numpy matplotlib seaborn pyyaml
```

---

### 7.5 Lỗi: GPU Not Found (Không phải lỗi nghiêm trọng)

**Triệu chứng:**
```
⚠️  No GPU found. Training will use CPU (slower).
```

**Giải pháp:**
- Không cần làm gì, vẫn train được
- Chỉ chậm hơn 3-4 lần
- Nếu muốn dùng GPU: Cài CUDA và cuDNN (phức tạp)

---

### 7.6 Lỗi: Jupyter Kernel Died

**Triệu chứng:**
```
The kernel appears to have died. It will restart automatically.
```

**Nguyên nhân:** RAM không đủ hoặc code có bug

**Giải pháp:**
1. Restart kernel: `Kernel` → `Restart`
2. Giảm sample_fraction xuống 0.05 (5%)
3. Chạy lại từ đầu

---

## 8. KIỂM TRA KẾT QUẢ

### 8.1 Checklist Hoàn Thành

- [ ] **Data Preprocessing:**
  - [ ] 6 files `.npz` trong `Output/data/`
  - [ ] 3 files artifacts trong `Output/models/`

- [ ] **Training:**
  - [ ] `global_model.h5` trong `Output/models/`
  - [ ] `training_history.json` trong `Output/metrics/`
  - [ ] Final accuracy > 95%

- [ ] **Evaluation:**
  - [ ] 3 PNG files (confusion matrix, accuracy plot, F1 chart)
  - [ ] `metrics_report.json`
  - [ ] `classification_report.txt`

---

### 8.2 Kiểm Tra Model Quality

**Accuracy:**
```
✅ Overall accuracy > 95%
```

**F1-Score:**
```
✅ All 34 classes have F1-Score > 0.85
```

**Confusion Matrix:**
```
✅ Diagonal có giá trị cao (correct predictions)
✅ Off-diagonal có giá trị thấp (misclassifications)
```

---

### 8.3 Xem Kết Quả

**Mở visualizations:**
```bash
open ../Output/metrics/confusion_matrix.png
open ../Output/metrics/accuracy_plot.png
open ../Output/metrics/f1_scores_per_class.png
```

**Đọc metrics:**
```bash
cat ../Output/metrics/classification_report.txt
```

---

## 9. TIPS & BEST PRACTICES

### 9.1 Lần Đầu Chạy

✅ **Luôn test với 10% data trước:**
```yaml
experimental:
  use_sample_data: true
  sample_fraction: 0.1
```

✅ **Chạy vào ban đêm hoặc cuối tuần:**
- Training full data mất 5-7 giờ
- Để máy chạy qua đêm

✅ **Không tắt máy khi đang training:**
- Sẽ mất hết tiến độ
- Phải chạy lại từ đầu

---

### 9.2 Monitoring

✅ **Theo dõi accuracy sau mỗi 5 rounds:**
- Nếu tăng đều → OK
- Nếu không tăng → Có vấn đề

✅ **Check RAM usage:**
```bash
# macOS
top -l 1 | grep PhysMem

# Linux
free -h
```

✅ **Check disk space:**
```bash
df -h
```

---

### 9.3 Backup

✅ **Backup model sau khi train xong:**
```bash
cp -r Output/ Output_backup_$(date +%Y%m%d)/
```

✅ **Backup training history:**
```bash
cp Output/metrics/training_history.json training_history_$(date +%Y%m%d).json
```

---

### 9.4 Optimization

**Nếu muốn train nhanh hơn:**

1. **Giảm num_rounds:**
   ```yaml
   num_rounds: 20  # Thay vì 30
   ```

2. **Giảm local_epochs:**
   ```yaml
   local_epochs: 3  # Thay vì 5
   ```

3. **Tăng batch_size (nếu RAM đủ):**
   ```yaml
   batch_size: 512  # Thay vì 256
   ```

**⚠️ Lưu ý:** Có thể giảm accuracy!

---

### 9.5 Troubleshooting Nhanh

| Vấn đề | Giải pháp |
|--------|-----------|
| OOM | Giảm chunk_size hoặc dùng sample |
| Training chậm | Dùng GPU hoặc giảm data |
| Accuracy thấp | Tăng rounds hoặc giảm learning_rate |
| Kernel died | Restart và giảm sample_fraction |

---

## 10. KẾT LUẬN

### 10.1 Tổng Kết

Sau khi hoàn thành 3 notebooks, bạn sẽ có:

✅ **Model trained** với accuracy > 95%  
✅ **Metrics chi tiết** cho tất cả 34 classes  
✅ **Visualizations đẹp** cho báo cáo đồ án  
✅ **Artifacts đầy đủ** cho Web App integration  

---

### 10.2 Thời Gian Dự Kiến

| Giai đoạn | 10% Data | 100% Data (GPU) | 100% Data (CPU) |
|-----------|----------|-----------------|-----------------|
| Data Preprocessing | 5-10 mins | 30-60 mins | 30-60 mins |
| FL Training | 30-60 mins | 4-6 hours | 8-12 hours |
| Evaluation | 5 mins | 10-20 mins | 10-20 mins |
| **TOTAL** | **40-75 mins** | **5-7 hours** | **9-13 hours** |

---

### 10.3 Next Steps

1. ✅ Review tất cả visualizations
2. ✅ Đưa plots vào báo cáo đồ án
3. ✅ Chuẩn bị demo cho giảng viên
4. ✅ Bắt đầu xây dựng Web App (sử dụng model đã train)

---

## 📞 HỖ TRỢ

Nếu gặp vấn đề:

1. **Đọc lại phần Troubleshooting** (Section 7)
2. **Check logs** trong notebook cells
3. **Google error message** cụ thể
4. **Hỏi giảng viên** hoặc bạn bè

---

**Chúc bạn training thành công! 🚀**

---

**Ngày cập nhật:** 28/12/2025  
**Phiên bản:** 1.0  
**Tác giả:** Nguyễn Đức Thắng
