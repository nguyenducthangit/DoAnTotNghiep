# Design: Federated Learning Training Infrastructure

## Context

Đồ án tốt nghiệp cần xây dựng một hệ thống Federated Learning để phát hiện tấn công mạng IoT. Hiện tại có:
- **Dataset**: CICIoT2023 (~12GB, 169 CSV files, 34 attack classes)
- **Existing work**: Một số notebooks thử nghiệm rời rạc
- **Goal**: Training model đạt >95% accuracy, xuất artifacts cho Web App

**Constraints**:
- Hardware: Laptop/desktop thông thường (16GB RAM, có thể có/không có GPU)
- Time: Cần hoàn thành trong vài tuần
- Deployment target: Web application sẽ load model để dự đoán real-time traffic

**Stakeholders**:
- Sinh viên (developer/researcher)
- Giảng viên hướng dẫn (reviewer)
- Future: End-users của Web App cảnh báo

## Goals / Non-Goals

### Goals
- ✅ Tạo ra một pipeline rõ ràng, dễ chạy lại (reproducible) cho FL training
- ✅ Đạt performance tiêu chuẩn (>95% accuracy, F1 >0.85 cho tất cả classes)
- ✅ Xuất các artifacts đầy đủ cho Web integration
- ✅ Tạo visualizations đẹp để đưa vào báo cáo đồ án
- ✅ Code sạch, có comments, dễ hiểu cho giảng viên review

### Non-Goals
- ❌ Không cần distributed training thực sự (chỉ simulation trên 1 máy)
- ❌ Không cần privacy mechanisms phức tạp (differential privacy, secure aggregation)
- ❌ Không cần real-time training hoặc online learning
- ❌ Không cần deploy model lên cloud (chỉ cần local files)

## Decisions

### Decision 1: Dataset Loading Strategy - Chunking vs In-Memory
**Choice**: Use **pandas chunking** with batch processing

**Why**:
- 12GB dataset không fit vào RAM nếu load toàn bộ
- Chunking cho phép process từng phần, tránh OOM
- Có thể downsample nếu cần (e.g., lấy 10% data để test nhanh)

**Alternatives considered**:
- ❌ Dask: Quá phức tạp cho use case này, overkill
- ❌ Spark: Cần setup cluster, không phù hợp với laptop
- ❌ SQLite: Thêm overhead, không cần truy vấn phức tạp

**Implementation**:
```python
chunks = []
for chunk in pd.read_csv(file, chunksize=50000):
    # Process chunk (clean, filter)
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)
```

### Decision 2: FL Simulation - Multi-process vs Single-process
**Choice**: **Single-process sequential** simulation

**Why**:
- Đơn giản hơn, dễ debug
- Không cần IPC (inter-process communication) phức tạp
- Training time chấp nhận được (~4-6 hours cho 50 rounds)
- Tránh race conditions và synchronization issues

**Alternatives considered**:
- ❌ Multi-threading: Python GIL làm giảm hiệu quả
- ❌ Multi-processing: Phức tạp, cần serialize model nhiều lần, overhead lớn

**Implementation**:
```python
for round in range(num_rounds):
    client_weights = []
    for client_id in range(num_clients):
        # Load client data
        # Train locally
        # Collect weights
        client_weights.append(local_weights)
    # Aggregate
    global_weights = fedavg(client_weights)
```

### Decision 3: Data Partitioning - IID vs Non-IID
**Choice**: **Non-IID** with label-based stratification

**Why**:
- Realistic: Các IoT devices thực tế có attack patterns khác nhau
- Challenge: Kiểm tra xem FL có handle được data heterogeneity không
- Research value: Có thể so sánh IID vs Non-IID trong báo cáo

**Approach**:
- Client 1: Nhiều DDoS attacks
- Client 2: Nhiều Reconnaissance attacks
- Client 3: Nhiều Web attacks
- Client 4: Nhiều Mirai attacks
- Client 5: Mix của tất cả + benign traffic

**Code sketch**:
```python
def partition_noniid(df, num_clients=5):
    # Group by attack types
    client_data = {i: [] for i in range(num_clients)}
    
    # Assign majority of DDoS to client 0
    ddos_data = df[df['label'].str.contains('DDoS')]
    client_data[0].append(ddos_data.sample(frac=0.7))
    
    # ... similar for other clients
    
    # Distribute remaining data evenly
    # ...
    
    return client_data
```

### Decision 4: Model Architecture - DNN vs CNN vs LSTM
**Choice**: **Deep Neural Network (DNN)** with 3 hidden layers

**Why**:
- Tabular data (network features) → DNN là standard choice
- CNN: Tốt cho spatial data (images), không cần thiết ở đây
- LSTM: Tốt cho time series, nhưng data này là per-packet features (không có temporal dependency rõ ràng)
- DNN đơn giản, train nhanh, performance tốt cho classification task

**Architecture**:
```
Input (46) → Dense(128, ReLU, Dropout=0.3) 
          → Dense(64, ReLU, Dropout=0.3)
          → Dense(32, ReLU, Dropout=0.3)
          → Dense(34, Softmax)
```

**Why this architecture**:
- Progressive dimension reduction: 46 → 128 → 64 → 32 → 34
- Dropout 0.3: Prevent overfitting (large dataset)
- Softmax: Multi-class classification (34 classes)

**Alternatives considered**:
- ❌ Shallow network (1-2 layers): Có thể underfitting với 34 classes
- ❌ Very deep (5+ layers): Overkill, slower training, risk overfitting
- ❌ CNN 1D: Thử nghiệm cho thấy không cải thiện accuracy đáng kể

### Decision 5: Aggregation Algorithm - FedAvg vs FedProx vs FedNova
**Choice**: **FedAvg** (Federated Averaging)

**Why**:
- Baseline standard trong FL research
- Simple và well-understood
- Performance tốt với Non-IID data (đã validated trong nhiều papers)
- Easy to implement: Chỉ cần average weights

**Alternatives considered**:
- ❌ FedProx: Thêm proximal term để handle heterogeneity, nhưng phức tạp hơn, cần tune thêm hyperparameter (μ)
- ❌ FedNova: Normalize updates theo số local steps, không cần thiết khi local epochs đồng nhất

**Implementation**:
```python
def fedavg(client_weights):
    avg_weights = []
    for layer in zip(*client_weights):
        avg_weights.append(np.mean(layer, axis=0))
    return avg_weights
```

### Decision 6: Output Format - H5 vs SavedModel vs ONNX
**Choice**: **Keras H5** format

**Why**:
- Single file, dễ transfer
- Web App có thể load bằng TensorFlow.js hoặc Flask backend với Keras
- Nhỏ gọn (~10MB), fast loading
- Compatible với `model.predict()` trực tiếp

**Alternatives considered**:
- ❌ SavedModel: Directory structure, phức tạp hơn
- ❌ ONNX: Cần thêm bước convert, không cần thiết vì không deploy lên edge devices

## Risks / Trade-offs

### Risk 1: Memory Overflow khi load dataset
**Mitigation**:
- Chunking strategy với `chunksize=50000`
- Nếu vẫn bị OOM, downsample xuống 10-20% data
- Monitor memory usage với `psutil`
- Document hardware requirements trong README

### Risk 2: Training không converge (do Non-IID data)
**Mitigation**:
- Tăng `num_rounds` từ 30 lên 50 nếu cần
- Plot accuracy curve sau mỗi round để monitor
- Fallback: Chuyển sang IID partitioning nếu Non-IID không work

### Risk 3: Model overfitting
**Mitigation**:
- Dropout layers (0.3)
- Early stopping nếu validation loss tăng
- Monitor training vs test accuracy gap

### Risk 4: Imbalanced classes
**Observation**: CICIoT2023 có class imbalance (benign traffic >> attack traffic)

**Mitigation**:
- Use `class_weight` trong model.fit() để balance loss
- Report per-class metrics (không chỉ overall accuracy)
- F1-Score là primary metric (hơn là accuracy)

### Trade-off 1: Training Time vs Accuracy
- **Choice**: Ưu tiên accuracy >95%, chấp nhận training time 4-6 hours
- **Rationale**: Đây là offline training, chỉ chạy 1 lần, có thể để qua đêm
- **Alternative**: Reduce num_rounds hoặc local_epochs để train nhanh hơn, nhưng có thể giảm accuracy

### Trade-off 2: Model Complexity vs Interpretability
- **Choice**: DNN với 3 hidden layers (moderate complexity)
- **Rationale**: Cần balance giữa performance và khả năng giải thích
- **Alternative**: Simpler model (Logistic Regression) dễ interpret hơn nhưng accuracy thấp hơn

## Migration Plan

### Phase 1: Setup (Day 1)
1. Create directory structure (`utils/`, `configs/`, `Output/`)
2. Create `training_config.yaml`
3. Install dependencies

### Phase 2: Data Preprocessing (Day 2-3)
1. Implement `data_utils.py`
2. Run `1_Data_Preprocessing.ipynb`
3. Verify partitioned data in `Output/data/`

### Phase 3: Model Training (Day 4-5)
1. Implement `model_utils.py` and `fl_utils.py`
2. Run `2_Federated_Training.ipynb`
3. Monitor training curves
4. Verify model saved to `Output/models/`

### Phase 4: Evaluation (Day 6)
1. Run `3_Model_Evaluation_Export.ipynb`
2. Generate all visualizations
3. Verify all metrics meet requirements (>95% accuracy)

### Phase 5: Validation (Day 7)
1. Test inference with sample data
2. Verify all exported files loadable
3. Write documentation

### Rollback Plan
Nếu FL training không work:
- **Fallback 1**: Centralized training (gộp tất cả data lại, train single model)
- **Fallback 2**: Reduce num_classes (34 → 8 classes hoặc 2 classes binary)
- **Fallback 3**: Use pre-trained model từ research papers

## Open Questions

1. **Q**: Có nên thử multiple model architectures (DNN vs CNN 1D vs Ensemble)?
   **A**: Bắt đầu với DNN, nếu có thời gian thì experiment thêm

2. **Q**: Có cần implement differential privacy?
   **A**: Non-goal cho MVP, có thể thêm sau nếu giảng viên yêu cầu

3. **Q**: Dataset có cần augmentation?
   **A**: Không, IoT network traffic không phù hợp với data augmentation

4. **Q**: Có nên split notebook ra nhiều files Python scripts?
   **A**: Giữ notebooks để dễ visualize và demo, nhưng logic chính trong `utils/` files

## References

- FedAvg paper: McMahan et al. (2017) - Communication-Efficient Learning of Deep Networks from Decentralized Data
- CICIoT2023 dataset: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- TensorFlow Federated: https://www.tensorflow.org/federated (reference only, không dùng trong project)
