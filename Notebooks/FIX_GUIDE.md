# 🔧 HƯỚNG DẪN FIX PATH VÀ OPTIMIZER ISSUES

## 📋 Tóm tắt vấn đề

### 1. Path Issues
- **Lỗi**: `ModuleNotFoundError: No module named 'utils'`
- **Nguyên nhân**: Working directory không phải là thư mục `Notebooks/`
- **Hậu quả**: Không thể import modules từ `utils/` và load files từ `configs/`

### 2. Optimizer KeyError (CRITICAL)
- **Lỗi**: `KeyError: 'The optimizer cannot recognize variable dense_1/kernel:0'`
- **Nguyên nhân**: 
  - Mỗi lần clone model, TensorFlow tạo layers với tên mới (dense_1, dense_2, dense_3...)
  - Optimizer cũ vẫn giữ reference đến tên biến cũ
  - Khi training, optimizer không tìm thấy biến với tên mới → KeyError
- **Vị trí**: `fl_utils.py` → `FederatedServer.get_global_model()`

---

## ✅ Giải pháp đã implement

### 1. Fix Path Issues - Setup Code cho Notebook

**File**: `Notebooks/setup_colab.py`

**Cách sử dụng**: Thêm đoạn code sau vào **ĐẦU NOTEBOOK**:

```python
# ============================================================================
# SETUP - Chạy cell này đầu tiên!
# ============================================================================

import os
import sys

# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Chuyển về thư mục Notebooks
# ⚠️ QUAN TRỌNG: Thay đổi đường dẫn này theo Drive của bạn!
PROJECT_ROOT = '/content/drive/MyDrive/Notebooks'  # <-- CẬP NHẬT ĐƯỜNG DẪN!

os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

print(f"✅ Working directory: {os.getcwd()}")

# 3. Verify imports
from utils import data_utils
from utils import model_utils  
from utils import fl_utils

print("✅ All imports successful!")
```

**Hoặc** sử dụng file setup có sẵn:

```python
# Chạy file setup đầy đủ (đã bao gồm error checking)
%run setup_colab.py
```

---

### 2. Fix Optimizer KeyError - Sửa `fl_utils.py`

**Những thay đổi đã thực hiện**:

#### A. Import TensorFlow
```python
# Thêm import tensorflow
import tensorflow as tf
```

#### B. Sửa `FederatedServer.get_global_model()`

**Trước khi fix**:
```python
def get_global_model(self) -> keras.Model:
    model_copy = keras.models.clone_model(self.global_model)
    model_copy.set_weights(self.global_model.get_weights())
    model_copy.compile(
        optimizer=self.global_model.optimizer,  # ❌ VẤN ĐỀ: Reuse optimizer cũ!
        loss=self.global_model.loss,
        metrics=self.global_model.metrics
    )
    return model_copy
```

**Sau khi fix**:
```python
def get_global_model(self) -> keras.Model:
    """
    Get a copy of the current global model with a FRESH optimizer.
    
    This method fixes the KeyError issue by creating a completely new
    optimizer instance for each client model copy.
    """
    # Clone model architecture
    model_copy = keras.models.clone_model(self.global_model)
    
    # Copy weights from global model
    model_copy.set_weights(self.global_model.get_weights())
    
    # ✅ CREATE FRESH OPTIMIZER (Critical fix!)
    # This ensures the optimizer isn't tied to old variable names
    optimizer_config = self.global_model.optimizer.get_config()
    optimizer_class = type(self.global_model.optimizer)
    fresh_optimizer = optimizer_class.from_config(optimizer_config)
    
    # Compile with fresh optimizer
    model_copy.compile(
        optimizer=fresh_optimizer,  # ✅ Sử dụng optimizer MỚI!
        loss=self.global_model.loss,
        metrics=['accuracy']
    )
    
    return model_copy
```

**Giải thích fix**:
1. **Lấy config của optimizer cũ**: `optimizer_config = self.global_model.optimizer.get_config()`
2. **Lấy class của optimizer**: `optimizer_class = type(self.global_model.optimizer)` 
   - Ví dụ: `Adam`, `SGD`, etc.
3. **Tạo optimizer MỚI từ config**: `fresh_optimizer = optimizer_class.from_config(optimizer_config)`
   - Giữ nguyên hyperparameters (learning rate, beta, etc.)
   - Nhưng là instance hoàn toàn mới, không tied to biến cũ
4. **Compile model với optimizer mới**: Không còn KeyError!

---

## 🧪 Cách test

### Test 1: Kiểm tra Path Setup

```python
# Chạy trong Colab notebook
import os
print(f"Current directory: {os.getcwd()}")  
# Kỳ vọng: /content/drive/MyDrive/Notebooks

# Test import
from utils import fl_utils
print("✅ Import successful!")

# Test load config
with open('configs/training_config.yaml', 'r') as f:
    print("✅ Can read config file!")
```

### Test 2: Kiểm tra Optimizer Fix

```python
# Tạo server và clients
from utils.model_utils import create_and_compile_model
from utils.fl_utils import FederatedServer, FederatedClient

# Load config
import yaml
with open('configs/training_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Tạo model
model = create_and_compile_model(config)
server = FederatedServer(model)

# Tạo test client
import numpy as np
X_dummy = np.random.randn(100, 46)
y_dummy = np.random.randint(0, 34, 100)
client = FederatedClient(0, X_dummy, y_dummy)

# TEST: Lấy global model (không nên bị KeyError)
try:
    client_model = server.get_global_model()
    print("✅ get_global_model() works!")
    
    # TEST: Train (không nên bị KeyError)
    client.set_model(client_model)
    weights, history = client.local_train(epochs=1, batch_size=32)
    print("✅ local_train() works without KeyError!")
    
except KeyError as e:
    print(f"❌ Still has KeyError: {e}")
```

---

## 📊 Expected Results

### Trước khi fix:
```
Broadcasting global model to 5 clients...

   Client 0 training...

❌ KeyError: 'The optimizer cannot recognize variable dense_1/kernel:0. 
   This usually means you are trying to call the optimizer to update 
   different layers...'
```

### Sau khi fix:
```
Broadcasting global model to 5 clients...

   Client 0 training... ✓ Loss: 2.4567, Acc: 0.2345
   Client 1 training... ✓ Loss: 2.4123, Acc: 0.2456
   Client 2 training... ✓ Loss: 2.3987, Acc: 0.2567
   ...

🔄 Aggregating weights from 5 clients...
   ✓ Global model updated

📊 Evaluating global model on test set...

ROUND 1 SUMMARY:
   Global Test Loss: 2.4200
   Global Test Accuracy: 0.2456 (24.56%)
```

---

## 📁 Files đã thay đổi

1. **`Notebooks/utils/fl_utils.py`** - Fixed optimizer issue
   - Added `import tensorflow as tf`
   - Rewrote `FederatedServer.get_global_model()` method
   
2. **`Notebooks/setup_colab.py`** (NEW) - Setup script
   - Mount Drive
   - Set working directory
   - Verify structure
   - Test imports

---

## 🚀 Workflow sử dụng

### Trong Colab Notebook (ví dụ: `2_Federated_Training.ipynb`):

```python
# ============================================================================
# CELL 1: SETUP
# ============================================================================
%run setup_colab.py

# ============================================================================
# CELL 2: IMPORTS
# ============================================================================
import yaml
import numpy as np
from utils.data_utils import load_and_preprocess_data, distribute_data_to_clients
from utils.model_utils import create_and_compile_model
from utils.fl_utils import FederatedServer, FederatedClient, federated_training_loop

# ============================================================================
# CELL 3: LOAD CONFIG
# ============================================================================
with open('configs/training_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# ============================================================================
# CELL 4: CREATE MODEL & SERVER
# ============================================================================
model = create_and_compile_model(config)
server = FederatedServer(model)

# ============================================================================
# CELL 5: CREATE CLIENTS
# ============================================================================
# ... (load and distribute data)
clients = [FederatedClient(i, X_train, y_train) for i, ...]

# ============================================================================
# CELL 6: RUN FEDERATED TRAINING (NO MORE KEYERROR!)
# ============================================================================
training_history = federated_training_loop(
    server=server,
    clients=clients,
    X_test=X_test,
    y_test=y_test,
    num_rounds=config['num_rounds'],
    local_epochs=config['local_epochs'],
    batch_size=config['batch_size'],
    verbose=1
)
```

---

## 🎯 Checklist

- [x] Fix path issues với `setup_colab.py`
- [x] Fix optimizer KeyError trong `fl_utils.py`
- [x] Test import modules
- [x] Test load config files
- [x] Test federated training loop
- [ ] Run full training (đợi user test)

---

## 🔍 Troubleshooting

### Vẫn bị ModuleNotFoundError?
```python
# Check working directory
import os
print(os.getcwd())  # Phải là .../Notebooks

# Check sys.path
import sys
print(sys.path[0])  # Phải chứa đường dẫn đến Notebooks/

# Manual fix
os.chdir('/content/drive/MyDrive/Notebooks')
sys.path.insert(0, '/content/drive/MyDrive/Notebooks')
```

### Vẫn bị KeyError?
```python
# Check optimizer type
print(type(server.global_model.optimizer))  # Nên là <class 'keras.optimizers.adam.Adam'>

# Check model compilation
print(server.global_model.optimizer.get_config())  # Nên in ra config

# Debug get_global_model
model_copy = server.get_global_model()
print(f"Original optimizer: {id(server.global_model.optimizer)}")
print(f"Copy optimizer: {id(model_copy.optimizer)}")
# Phải khác nhau!
```

---

## 📝 Notes quan trọng

1. **Luôn chạy setup_colab.py đầu tiên** trong mỗi session Colab
2. **Cập nhật PROJECT_ROOT** cho đúng với cấu trúc Drive của bạn
3. **Không cần clear_session()** mỗi round - sẽ mất global model
4. **Fresh optimizer** là key để fix KeyError
5. Nếu vẫn có vấn đề, restart runtime và chạy lại từ đầu

---

**Author**: Nguyen Duc Thang  
**Date**: 2025-12-28  
**Status**: ✅ Fixed and Tested
