# TÀI LIỆU GIẢI THÍCH CHI TIẾT: NOTEBOOK TIỀN XỬ LÝ DỮ LIỆU
**File gốc:** `1_Data_Preprocessing.ipynb`  
**Tác giả tài liệu:** AI Assistant  
**Ngày tạo:** 30/12/2025

---

## MỤC LỤC
1. [Tổng quan về Notebook](#1-tổng-quan-về-notebook)
2. [Phần 1: Setup và Import thư viện](#2-phần-1-setup-và-import-thư-viện)
3. [Phần 2: Tải cấu hình](#3-phần-2-tải-cấu-hình)
4. [Phần 3: Tải và Tiền xử lý Dataset](#4-phần-3-tải-và-tiền-xử-lý-dataset)
5. [Phần 4: Lọc đặc trưng bằng GSA](#5-phần-4-lọc-đặc-trưng-bằng-gsa)
6. [Phần 5: Mã hóa nhãn](#6-phần-5-mã-hóa-nhãn)
7. [Phần 6: Chuẩn hóa Features](#7-phần-6-chuẩn-hóa-features)
8. [Phần 7: Phân chia dữ liệu cho FL](#8-phần-7-phân-chia-dữ-liệu-cho-federated-learning)
9. [Phần 8-10: Lưu trữ và Verification](#9-phần-8-10-lưu-trữ-và-verification)

---

## 1. TỔNG QUAN VỀ NOTEBOOK

### 1.1 Mục tiêu chính
Notebook này là **bước đầu tiên** trong pipeline huấn luyện mô hình Federated Learning để phát hiện tấn công mạng IoT. Nó thực hiện:

1. **Tải dữ liệu khổng lồ** (~12GB, 169 file CSV) từ bộ dataset CICIoT2023
2. **Làm sạch dữ liệu** (loại bỏ giá trị null, duplicate)
3. **Lọc đặc trưng** bằng thuật toán GSA (từ 46 → 22 features)
4. **Mã hóa nhãn** (chuyển 34 tên tấn công thành số 0-33)
5. **Chuẩn hóa** tất cả features về khoảng [0, 1]
6. **Phân chia dữ liệu** cho 5 máy khách (clients) theo chiến lược Non-IID

### 1.2 Đầu ra (Outputs) quan trọng
Sau khi chạy xong, bạn sẽ có:
- `client_0_data.npz` đến `client_4_data.npz`: Dữ liệu cho 5 clients
- `test_data.npz`: Dữ liệu test chung
- `scaler.pkl`: Bộ chuẩn hóa (MinMaxScaler)
- `label_encoder.pkl` & `labels.json`: Bảng mã hóa nhãn

---

## 2. PHẦN 1: SETUP VÀ IMPORT THƯ VIỆN

### 2.1 Code gốc
```python
# Standard libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path

# Import our utility modules
from utils import data_utils
from utils import fl_utils_pytorch
from utils.includes import X_columns, y_column, dict_34_classes

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

# Set random seed for reproducibility
np.random.seed(42)
```

### 2.2 Giải thích chi tiết

#### A. Thư viện chuẩn (Standard Libraries)
- **`os`, `sys`**: Quản lý đường dẫn file, thư mục
- **`numpy`**: Tính toán số học, xử lý ma trận
- **`pandas`**: Xử lý dữ liệu dạng bảng (DataFrame)
- **`matplotlib`, `seaborn`**: Vẽ biểu đồ, trực quan hóa
- **`yaml`**: Đọc file cấu hình `.yaml`
- **`pathlib.Path`**: Xử lý đường dẫn hiện đại hơn `os.path`

#### B. Module tự viết (Custom Utilities)
- **`data_utils`**: Chứa các hàm:
  - `clean_data()`: Loại bỏ null, duplicate
  - `encode_labels()`: Chuyển tên tấn công → số
  - `normalize_features()`: Chuẩn hóa về [0,1]
  - `partition_data_noniid()`: Chia dữ liệu Non-IID
  
- **`fl_utils_pytorch`**: Các hàm hỗ trợ Federated Learning (sẽ dùng ở Notebook 2)

- **`utils.includes`**: Định nghĩa:
  - `X_columns`: Danh sách 46 tên cột đặc trưng
  - `y_column`: Tên cột nhãn (`'label'`)
  - `dict_34_classes`: Dictionary ánh xạ 34 loại tấn công

#### C. Cấu hình hiển thị
```python
pd.set_option('display.max_columns', 50)  # Hiển thị tối đa 50 cột
pd.set_option('display.max_rows', 100)    # Hiển thị tối đa 100 dòng
```
**Tại sao cần?** Dataset có 47 cột, nếu không set thì Pandas sẽ ẩn bớt cột khi in ra.

#### D. Random Seed
```python
np.random.seed(42)
```
**Tại sao cần?** Đảm bảo kết quả **có thể tái tạo** (reproducible). Mỗi lần chạy code sẽ cho kết quả giống nhau.

### 2.3 Output khi chạy
```
✅ All imports successful!
   Number of features: 46
   Label column: label
   Number of attack classes: 34
```

---

## 3. PHẦN 2: TẢI CẤU HÌNH

### 3.1 Code gốc
```python
# Load training configuration
config_path = 'configs/training_config.yaml'

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print("📄 Configuration loaded:")
print(f"   Number of clients: {config['num_clients']}")
print(f"   Test split ratio: {config['data']['test_split_ratio']}")
print(f"   Chunk size: {config['data']['chunk_size']}")
print(f"   Partition strategy: {config['data']['partition_strategy']}")
print(f"   Use sample data: {config['experimental']['use_sample_data']}")
```

### 3.2 Giải thích chi tiết

#### A. File `training_config.yaml` chứa gì?
Đây là file cấu hình tập trung, chứa tất cả tham số quan trọng:

```yaml
num_clients: 5                    # Số lượng máy khách (clients)
random_seed: 42                   # Seed cho random

data:
  test_split_ratio: 0.2           # 20% dữ liệu dùng để test
  chunk_size: 50000               # Đọc 50,000 dòng một lúc (tiết kiệm RAM)
  partition_strategy: 'non_iid'   # Chiến lược chia dữ liệu

experimental:
  use_sample_data: false          # Có dùng dữ liệu mẫu nhỏ không?
  sample_fraction: 0.05           # Nếu có, lấy 5% dữ liệu
```

#### B. Tại sao dùng file YAML thay vì hard-code?
**Ưu điểm:**
1. **Dễ thay đổi**: Không cần sửa code, chỉ cần sửa file YAML
2. **Tái sử dụng**: Có thể tạo nhiều file config khác nhau cho các thí nghiệm
3. **Dễ đọc**: Cú pháp YAML rất dễ hiểu

#### C. Các tham số quan trọng

**`num_clients: 5`**
- Chia dữ liệu cho **5 máy khách** (mô phỏng 5 thiết bị IoT khác nhau)
- Trong thực tế có thể là 10, 20, 100 clients

**`test_split_ratio: 0.2`**
- 20% dữ liệu dùng để **test** (đánh giá cuối cùng)
- 80% còn lại dùng để **train** (chia cho 5 clients)

**`chunk_size: 50000`**
- Khi đọc file CSV lớn, chỉ đọc 50,000 dòng một lúc vào RAM
- **Tại sao?** Tránh tràn RAM khi file quá lớn (12GB)

**`partition_strategy: 'non_iid'`**
- **IID** (Independent and Identically Distributed): Mỗi client có dữ liệu giống nhau
- **Non-IID**: Mỗi client có dữ liệu **khác nhau** (thực tế hơn)
- Ví dụ: Client 1 chủ yếu thấy tấn công DDoS, Client 2 chủ yếu thấy Malware

### 3.3 Output khi chạy
```
📄 Configuration loaded:
   Number of clients: 5
   Test split ratio: 0.2
   Chunk size: 50000
   Partition strategy: non_iid
   Use sample data: False
```

---

## 4. PHẦN 3: TẢI VÀ TIỀN XỬ LÝ DATASET

### 4.1 Tổng quan chiến lược
Đây là phần **QUAN TRỌNG NHẤT** và cũng **PHỨC TẠP NHẤT** của Notebook. Code thực hiện:

1. **Kiểm tra Cache**: Nếu đã xử lý rồi → tải trực tiếp
2. **Nếu chưa có Cache**: Chạy pipeline đầy đủ:
   - Tải 169 file CSV
   - Merge thành 1 DataFrame
   - **SHUFFLE** (cực kỳ quan trọng!)
   - Chia Train/Test
   - Lưu vào Cache

### 4.2 Code phần CACHE HIT (Đã xử lý rồi)

```python
if os.path.exists(train_file) and os.path.exists(test_file):
    print("CACHE HIT: Loading preprocessed datasets from disk")
    
    df_train = pd.read_csv(train_file, low_memory=False)
    df_test = pd.read_csv(test_file, low_memory=False)
    
    print(f"✅ Loaded from cache successfully!")
    print(f"   Train shape: {df_train.shape}")
    print(f"   Test shape: {df_test.shape}")
```

#### Giải thích:
- **`train_file`**: `../Output/preprocessed/train_dataset.csv`
- **`test_file`**: `../Output/preprocessed/test_dataset.csv`
- **`low_memory=False`**: Đọc toàn bộ file vào RAM (nhanh hơn nhưng tốn RAM)

**Kết quả:**
```
Train shape: (2487431, 47)  # 2.4 triệu dòng, 47 cột
Test shape: (130917, 47)    # 130 nghìn dòng, 47 cột
```

### 4.3 Code phần CACHE MISS (Chưa xử lý)

#### BƯỚC 1: Tải tất cả file CSV

```python
csv_pattern = os.path.join(data_dir, '*.csv')
csv_files = sorted(glob.glob(csv_pattern))

dataframes = []
total_rows = 0

for i, file_path in enumerate(csv_files):
    try:
        df_file = pd.read_csv(file_path, low_memory=False)
        rows = len(df_file)
        total_rows += rows
        dataframes.append(df_file)
        
        if (i + 1) % 20 == 0:  # In progress mỗi 20 file
            print(f"[{i+1}/{len(csv_files)}] Loaded {os.path.basename(file_path)}")
    except Exception as e:
        print(f"⚠️ Error loading {file_path}: {e}")
        continue
```

**Giải thích từng dòng:**

1. **`glob.glob(csv_pattern)`**: Tìm tất cả file `.csv` trong thư mục `../DataTests`
2. **`sorted(...)`**: Sắp xếp theo tên file (đảm bảo thứ tự nhất quán)
3. **`dataframes.append(df_file)`**: Thêm DataFrame vào list
4. **`if (i + 1) % 20 == 0`**: Chỉ in thông báo mỗi 20 file (tránh spam console)
5. **`try...except`**: Nếu file bị lỗi, bỏ qua và tiếp tục

**Tại sao không dùng `pd.concat()` ngay?**
- Vì `concat()` tốn RAM. Tốt hơn là load hết vào list trước, rồi concat 1 lần.

#### BƯỚC 2: Merge tất cả DataFrame

```python
df_merged = pd.concat(dataframes, ignore_index=True)
print(f"✅ Merged DataFrame created!")
print(f"   Shape: {df_merged.shape}")
print(f"   Memory usage: {df_merged.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

del dataframes  # Giải phóng RAM
```

**Giải thích:**
- **`pd.concat(dataframes, ignore_index=True)`**: Ghép tất cả DataFrame thành 1
- **`ignore_index=True`**: Tạo lại index từ 0 (không giữ index cũ)
- **`del dataframes`**: Xóa biến để giải phóng RAM

#### BƯỚC 3: SHUFFLE (XÁO TRỘN) - CỰC KỲ QUAN TRỌNG!

```python
df_shuffled = df_merged.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
```

**Giải thích:**
- **`sample(frac=1.0)`**: Lấy mẫu 100% dữ liệu (tức là lấy hết, nhưng theo thứ tự ngẫu nhiên)
- **`random_state=42`**: Đảm bảo shuffle giống nhau mỗi lần chạy
- **`reset_index(drop=True)`**: Tạo lại index từ 0

---

## 5. PHẦN 4: LỌC ĐẶC TRƯNG BẰNG GSA (FEATURE SELECTION)

### 5.1 Tại sao cần lọc đặc trưng?

Dataset ban đầu có **46 đặc trưng** (features). Nhưng không phải tất cả đều hữu ích:
- Một số features có thể là **nhiễu** (noise) - không liên quan đến việc phân loại
- Một số features có thể **dư thừa** (redundant) - chứa thông tin trùng lặp
- Quá nhiều features → **Overfitting** (mô hình học thuộc lòng thay vì học quy luật)

**Giải pháp:** Sử dụng thuật toán **GSA (Gravitational Search Algorithm)** để tự động chọn ra **20 features tốt nhất**.

### 5.2 GSA là gì?

GSA (Gravitational Search Algorithm) là thuật toán tối ưu hóa lấy cảm hứng từ **định luật vạn vật hấp dẫn** của Newton.

**Ý tưởng cơ bản:**
1. Mỗi "hạt" (particle) đại diện cho một **bộ features** (ví dụ: chọn 20 trong 46 features)
2. "Khối lượng" của hạt = **Độ chính xác** khi dùng bộ features đó để phân loại
3. Các hạt có khối lượng lớn (độ chính xác cao) sẽ **hút** các hạt khác về phía mình
4. Sau nhiều vòng lặp, các hạt sẽ **hội tụ** về bộ features tốt nhất

### 5.3 Code chi tiết

```python
# Cấu hình GSA
gsa_config = {
    'enabled': True,
    'target_features': 20,      # Chọn 20 features
    'population_size': 15,      # 15 "hạt" trong quần thể
    'max_iterations': 30,       # Chạy tối đa 30 vòng lặp
    'sample_fraction': 0.05     # Dùng 5% dữ liệu để tăng tốc
}

# Chạy GSA
from utils.gsa_algorithm import GSA

gsa = GSA(
    n_features=len(X_columns),
    target_features=20,
    population_size=15,
    max_iterations=30
)

# Lấy mẫu 5% dữ liệu
sample_size = int(len(df_train) * 0.05)
df_sample = df_train.sample(n=sample_size, random_state=42)

X_sample = df_sample[X_columns].values
y_sample = df_sample[y_column].values

# Chạy tối ưu hóa
selected_indices = gsa.optimize(X_sample, y_sample)
selected_features = [X_columns[i] for i in selected_indices]
```

**Giải thích từng dòng:**

1. **`target_features=20`**: Mục tiêu là chọn 20 features tốt nhất
2. **`population_size=15`**: Có 15 "hạt" trong quần thể (mỗi hạt là 1 bộ 20 features)
3. **`max_iterations=30`**: Chạy tối đa 30 vòng lặp để tìm bộ features tốt nhất
4. **`sample_fraction=0.05`**: Chỉ dùng 5% dữ liệu để tăng tốc (vì GSA chạy rất lâu)

### 5.4 Kết quả

Sau khi chạy GSA, ta được:
- **22 features được chọn** (thay vì 20 như mục tiêu - có thể do thuật toán điều chỉnh)
- **Fitness/Accuracy: 0.9568** (95.68% độ chính xác trên tập mẫu)
- **Giảm 52.2% kích thước** (từ 46 → 22 features)

**Lợi ích:**
- Mô hình chạy **nhanh hơn** (ít features hơn)
- **Giảm overfitting** (loại bỏ nhiễu)
- **Tăng độ chính xác** (chỉ giữ lại features quan trọng)

---

## 6. PHẦN 5: MÃ HÓA NHÃN (ENCODE LABELS)

### 6.1 Tại sao cần mã hóa nhãn?

Máy tính **không hiểu chữ**, chỉ hiểu **số**. Trong dataset, cột `label` chứa tên các loại tấn công dưới dạng text:
- `"DDoS-ICMP_Flood"`
- `"BenignTraffic"`
- `"Mirai-greeth_flood"`
- ...

Ta cần chuyển chúng thành **số** để máy tính có thể xử lý.

### 6.2 Code chi tiết

```python
from sklearn.preprocessing import LabelEncoder
import json

# Tạo LabelEncoder
label_encoder = LabelEncoder()

# Mã hóa nhãn
df_train[y_column] = label_encoder.fit_transform(df_train[y_column])

# Lưu label encoder để dùng sau này
import pickle
with open('../Output/models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Lưu mapping (ánh xạ) từ số → tên
label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
with open('../Output/models/labels.json', 'w') as f:
    json.dump(label_mapping, f, indent=2)
```

**Giải thích từng bước:**

1. **`LabelEncoder()`**: Tạo bộ mã hóa nhãn
2. **`fit_transform()`**: Học và chuyển đổi nhãn thành số
   - Ví dụ: `"DDoS-ICMP_Flood"` → `6`
   - `"BenignTraffic"` → `1`
3. **Lưu `label_encoder.pkl`**: Để sau này có thể decode ngược lại
4. **Lưu `labels.json`**: File JSON dễ đọc, chứa mapping:
   ```json
   {
     "0": "Backdoor_Malware",
     "1": "BenignTraffic",
     "6": "DDoS-ICMP_Flood",
     ...
   }
   ```

### 6.3 Kết quả

Sau khi mã hóa:
- **34 loại tấn công** → **Số từ 0 đến 33**
- Cột `label` giờ chứa số thay vì text
- Ví dụ: `"DDoS-ICMP_Flood"` → `6`

---

## 7. PHẦN 6: CHUẨN HÓA FEATURES (NORMALIZE)

### 7.1 Tại sao cần chuẩn hóa?

Các features có **đơn vị khác nhau**:
- `flow_duration`: Có thể từ 0 đến vài triệu (microseconds)
- `fin_flag_number`: Chỉ có giá trị 0 hoặc 1
- `Rate`: Có thể từ 0 đến vài nghìn

Nếu không chuẩn hóa:
- Features có giá trị lớn sẽ **chi phối** quá trình học
- Mô hình Deep Learning sẽ **học chậm** hoặc **không hội tụ**

**Giải pháp:** Dùng **MinMaxScaler** để đưa tất cả về khoảng `[0, 1]`.

### 7.2 Code chi tiết

```python
from sklearn.preprocessing import MinMaxScaler

# Tạo scaler
scaler = MinMaxScaler()

# Chuẩn hóa features
df_train[X_columns] = scaler.fit_transform(df_train[X_columns])

# Lưu scaler để dùng cho dữ liệu mới
import pickle
with open('../Output/models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

**Giải thích:**

1. **`MinMaxScaler()`**: Tạo bộ chuẩn hóa
2. **`fit_transform()`**: Học min/max của mỗi feature và chuẩn hóa
   - Công thức: `X_scaled = (X - X_min) / (X_max - X_min)`
   - Kết quả: Tất cả giá trị nằm trong `[0, 1]`
3. **Lưu `scaler.pkl`**: **CỰC KỲ QUAN TRỌNG!**
   - Khi có dữ liệu mới (gói tin mạng mới), bạn **PHẢI** dùng scaler này để chuẩn hóa
   - Nếu không, mô hình sẽ dự đoán sai!

### 7.3 Ví dụ minh họa

**Trước khi chuẩn hóa:**
```
flow_duration: 4.888106
Rate: 0.409156
fin_flag_number: 0.0
```

**Sau khi chuẩn hóa:**
```
flow_duration: 0.0000488  (đã chia cho max)
Rate: 0.0000041
fin_flag_number: 0.0
```

---

## 8. PHẦN 7: PHÂN CHIA DỮ LIỆU CHO FEDERATED LEARNING

### 8.1 Chiến lược Non-IID

**IID (Independent and Identically Distributed):**
- Mỗi Client có dữ liệu **giống hệt nhau**
- Ví dụ: Client 1, 2, 3, 4, 5 đều có 20% mỗi loại tấn công

**Non-IID:**
- Mỗi Client có dữ liệu **khác nhau**
- Ví dụ:
  - Client 1: 80% DDoS, 20% Benign
  - Client 2: 90% Mirai, 10% DDoS
  - Client 3: 70% Benign, 30% Web attacks

**Tại sao dùng Non-IID?**
- Trong thực tế, mỗi thiết bị IoT chỉ thấy **một phần** lưu lượng mạng
- Mô phỏng đúng môi trường thực tế của Federated Learning

### 8.2 Code chi tiết

```python
def partition_data_noniid(df_train, num_clients=5, label_col='label', 
                          test_split=0.2, random_seed=42):
    # Chia train/test
    train_data, test_data = train_test_split(
        df_train, test_size=test_split, 
        stratify=df_train[label_col],  # Đảm bảo tỷ lệ nhãn đồng đều
        random_state=random_seed
    )
    
    # Sắp xếp theo nhãn
    train_data = train_data.sort_values(by=label_col)
    
    # Chia thành các "shard" (mảnh)
    num_shards = num_clients * 2
    shard_size = len(train_data) // num_shards
    
    # Gán shard cho từng client (mỗi client nhận 2 shards ngẫu nhiên)
    client_data = {}
    for i in range(num_clients):
        # Chọn 2 shards ngẫu nhiên
        shard_indices = np.random.choice(num_shards, 2, replace=False)
        
        # Lấy dữ liệu từ 2 shards
        client_df = pd.concat([
            train_data[shard_indices[0]*shard_size:(shard_indices[0]+1)*shard_size],
            train_data[shard_indices[1]*shard_size:(shard_indices[1]+1)*shard_size]
        ])
        
        client_data[f'client_{i}'] = {
            'X': client_df[X_columns].values,
            'y': client_df[label_col].values
        }
    
    # Test set chung
    client_data['test'] = {
        'X': test_data[X_columns].values,
        'y': test_data[label_col].values
    }
    
    return client_data
```

**Giải thích từng bước:**

1. **Chia Train/Test:**
   - 80% train, 20% test
   - `stratify`: Đảm bảo tỷ lệ các nhãn giống nhau ở train và test

2. **Sắp xếp theo nhãn:**
   - Để các nhãn giống nhau nằm gần nhau

3. **Chia thành Shards:**
   - Chia train thành `num_clients * 2 = 10` mảnh
   - Mỗi mảnh có kích thước bằng nhau

4. **Gán Shard cho Client:**
   - Mỗi Client nhận **2 shards ngẫu nhiên**
   - → Mỗi Client có phân phối nhãn khác nhau (Non-IID)

### 8.3 Kết quả

Sau khi phân chia:
```
Client 0: 397,989 samples
Client 1: 397,989 samples
Client 2: 397,989 samples
Client 3: 397,989 samples
Client 4: 397,988 samples
Test: 497,487 samples
```

---

## 9. PHẦN 8-10: LƯU TRỮ VÀ VERIFICATION

### 9.1 Lưu dữ liệu đã phân chia

```python
import numpy as np

output_dir = '../Output/data'
os.makedirs(output_dir, exist_ok=True)

for client_name, data in client_data.items():
    file_path = os.path.join(output_dir, f'{client_name}_data.npz')
    np.savez_compressed(
        file_path,
        X=data['X'],
        y=data['y']
    )
```

**Giải thích:**
- **`.npz`**: Format file binary của Numpy (nén, tải cực nhanh)
- Mỗi file chứa 2 arrays: `X` (features) và `y` (labels)

### 9.2 Verification (Kiểm tra)

```python
# Kiểm tra file đã lưu
for client_name in client_data.keys():
    file_path = os.path.join(output_dir, f'{client_name}_data.npz')
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path) / 1024**2  # MB
        print(f"✓ {client_name}_data.npz ({file_size:.2f} MB)")
```

---

## 10. TÓM TẮT TOÀN BỘ QUY TRÌNH

### Bước 1: Setup & Import
- Import thư viện chuẩn và module tự viết
- Set random seed để reproducible

### Bước 2: Load Configuration
- Đọc file `training_config.yaml`
- Lấy các tham số: số clients, test split ratio, ...

### Bước 3: Load & Preprocess Dataset
- **Kiểm tra Cache**: Nếu đã xử lý → tải trực tiếp
- **Nếu chưa có Cache:**
  1. Tải 169 file CSV
  2. Merge thành 1 DataFrame
  3. **SHUFFLE** (cực kỳ quan trọng!)
  4. Chia Train/Test (95%/5%)
  5. Lưu vào Cache

### Bước 4: GSA Feature Selection
- Dùng 5% dữ liệu để chạy GSA
- Chọn 22 features tốt nhất từ 46 features
- Giảm 52.2% kích thước

### Bước 5: Encode Labels
- Chuyển 34 tên tấn công → Số 0-33
- Lưu `label_encoder.pkl` và `labels.json`

### Bước 6: Normalize Features
- Dùng MinMaxScaler đưa tất cả về [0, 1]
- Lưu `scaler.pkl` để dùng cho dữ liệu mới

### Bước 7: Partition Data (Non-IID)
- Chia dữ liệu cho 5 Clients
- Mỗi Client có phân phối nhãn khác nhau
- Test set chung cho tất cả

### Bước 8: Save & Verify
- Lưu dữ liệu dưới dạng `.npz`
- Kiểm tra file đã lưu thành công

---

## 11. CÂU HỎI THƯỜNG GẶP (FAQ)

**Q1: Tại sao phải Shuffle dữ liệu?**
- A: Vì dữ liệu gốc có thể sắp xếp theo thời gian hoặc loại tấn công. Nếu không shuffle, train/test sẽ bị lệch.

**Q2: Tại sao chỉ dùng 5% dữ liệu cho GSA?**
- A: Vì GSA chạy rất lâu (30-60 phút). Dùng 5% vừa đủ để tìm features tốt mà không mất quá nhiều thời gian.

**Q3: File `scaler.pkl` dùng để làm gì?**
- A: Khi có gói tin mạng mới cần dự đoán, bạn PHẢI dùng scaler này để chuẩn hóa trước khi đưa vào mô hình.

**Q4: Tại sao dùng Non-IID thay vì IID?**
- A: Vì trong thực tế, mỗi thiết bị IoT chỉ thấy một phần lưu lượng mạng. Non-IID mô phỏng đúng môi trường thực tế.

**Q5: File `.npz` là gì?**
- A: Là format file binary của Numpy, cho phép lưu và tải dữ liệu cực nhanh.

---

> [!IMPORTANT]
> **Điểm quan trọng cần nhớ:**
> 1. **Shuffle** là bước KHÔNG THỂ thiếu
> 2. **GSA** giúp giảm 52% kích thước và tăng độ chính xác
> 3. **Scaler** phải được lưu lại để dùng cho dữ liệu mới
> 4. **Non-IID** mô phỏng đúng môi trường thực tế của FL

---

**HẾT PHẦN GIẢI THÍCH CHI TIẾT**

Nếu bạn có câu hỏi về bất kỳ phần nào, hãy hỏi tôi nhé!

**TẠI SAO PHẢI SHUFFLE?**

Đây là câu hỏi CỰC KỲ QUAN TRỌNG! Hãy hiểu kỹ:

**Trước khi shuffle:**
```
File 1: DDoS-UDP (10,000 dòng)
File 2: DDoS-TCP (15,000 dòng)
File 3: Benign (20,000 dòng)
File 4: Malware (8,000 dòng)
...
```

Nếu bạn **KHÔNG shuffle** mà chia luôn:
- **Train set** (95% đầu): Chủ yếu là DDoS-UDP, DDoS-TCP
- **Test set** (5% cuối): Chủ yếu là Malware, Benign

**Hậu quả:**
- Mô hình học trên DDoS → Test trên Malware → **Accuracy = 0%**!

**Sau khi shuffle:**
```
Dòng 1: Malware
Dòng 2: Benign
Dòng 3: DDoS-UDP
Dòng 4: Benign
Dòng 5: DDoS-TCP
...
```

Bây giờ Train và Test đều có **phân phối đồng đều** các loại tấn công → Mô hình học đúng!

#### BƯỚC 4: Chia Train/Test

```python
df_test = df_shuffled.head(int(len(df_shuffled) * test_size))
df_train = df_shuffled.tail(len(df_shuffled) - len(df_test))
```

**Giải thích:**
- **`test_size = 0.05`**: Lấy 5% đầu làm Test
- **`head()`**: Lấy n dòng đầu
- **`tail()`**: Lấy n dòng cuối

**Tại sao không dùng `train_test_split()` của sklearn?**
- Có thể dùng! Nhưng cách này đơn giản hơn và tiết kiệm RAM.

#### BƯỚC 5: Lưu vào Cache

```python
df_train.to_csv(train_file, index=False)
df_test.to_csv(test_file, index=False)
```

**Giải thích:**
- **`index=False`**: Không lưu cột index vào CSV
- Lần sau chạy lại → CACHE HIT → Nhanh hơn rất nhiều!

### 4.4 Kết quả cuối cùng

```
Train shape: (2,487,431, 47)  # 2.4 triệu dòng
Test shape: (130,917, 47)     # 130 nghìn dòng
```

**Phân tích:**
- **47 cột** = 46 features + 1 label
- **Tổng**: 2,618,348 dòng dữ liệu

---

## 5. PHẦN 4: LỌC ĐẶC TRƯNG BẰNG GSA

*(Phần này sẽ được viết tiếp...)*

---

**LƯU Ý:** Tài liệu này đang được viết. Tôi sẽ tiếp tục bổ sung các phần còn lại (GSA, Encode, Normalize, Partition). Bạn có thể đọc phần này trước và cho tôi biết có chỗ nào chưa rõ không nhé!
