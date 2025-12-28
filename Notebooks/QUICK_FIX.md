# 🚀 QUICK START - Fix Path & Optimizer Issues

## TL;DR - Làm gì ngay bây giờ?

### 1️⃣ Thêm code này vào ĐẦU notebook trong Colab:

```python
# =============================================================================
# SETUP CODE - Chạy cell này đầu tiên!
# =============================================================================
import os
import sys

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# ⚠️ THAY ĐỔI ĐƯỜNG DẪN NÀY CHO ĐÚNG!
PROJECT_ROOT = '/content/drive/MyDrive/Notebooks'  # <-- Sửa đường dẫn này!

# Chuyển directory
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

print(f"✅ Working in: {os.getcwd()}")

# Test import
from utils import data_utils, model_utils, fl_utils
print("✅ Imports successful!")
```

### 2️⃣ File `fl_utils.py` đã được fix tự động!

**Không cần làm gì thêm.** Optimizer KeyError đã được fix trong file:
- `/Notebooks/utils/fl_utils.py`

Thay đổi chính:
- ✅ Mỗi client nhận optimizer MỚI (không reuse optimizer cũ)
- ✅ Không còn KeyError: "optimizer cannot recognize variable dense_X/kernel"

---

## 📊 Chạy Training

```python
# Import
import yaml
from utils.model_utils import create_and_compile_model
from utils.fl_utils import FederatedServer, FederatedClient, federated_training_loop

# Load config
with open('configs/training_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create model & server
model = create_and_compile_model(config)
server = FederatedServer(model)

# ... (tạo clients như bình thường)

# Run training (NO MORE ERRORS!)
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

print("✅ Training completed successfully!")
```

---

## ⚠️ Lưu ý

1. **Cập nhật PROJECT_ROOT**: Đường dẫn phải trỏ đến thư mục `Notebooks/` trên Drive
2. **Restart runtime nếu gặp lỗi**: `Runtime > Restart runtime` trong Colab
3. **Chạy setup mỗi session**: Mỗi lần mở notebook phải chạy lại setup code

---

## 📁 Files đã fix

- ✅ `Notebooks/utils/fl_utils.py` - Fixed optimizer issue
- ✅ `Notebooks/setup_colab.py` - Complete setup script
- ✅ `Notebooks/FIX_GUIDE.md` - Detailed documentation

---

**Chi tiết đầy đủ**: Xem file `FIX_GUIDE.md`
