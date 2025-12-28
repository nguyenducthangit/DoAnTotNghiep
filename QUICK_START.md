# QUICK START GUIDE - Federated Learning Training

**⏱️ Thời gian đọc: 5 phút**

---

## 🚀 BẮT ĐẦU NHANH (3 BƯỚC)

### BƯỚC 1: Cài Đặt (5 phút)

```bash
# 1. Mở Terminal
cd "/Users/user/Documents/ĐỒ ÁN/Do an"

# 2. Cài packages
pip3 install tensorflow keras scikit-learn pandas numpy matplotlib seaborn pyyaml jupyter

# 3. Verify
python3 -c "import tensorflow; print('✅ Ready!')"
```

---

### BƯỚC 2: Cấu Hình (2 phút)

**⚠️ LẦN ĐẦU: Test với 10% data**

Mở file: `Notebooks/configs/training_config.yaml`

Sửa dòng:
```yaml
experimental:
  use_sample_data: true    # ← Đổi thành true
  sample_fraction: 0.1     # Dùng 10% data
```

---

### BƯỚC 3: Chạy Notebooks (40-75 phút với 10% data)

```bash
# Mở Jupyter
cd Notebooks
jupyter notebook
```

**Trong Jupyter, chạy lần lượt:**

1. **`1_Data_Preprocessing.ipynb`** (5-10 phút)
   - Chạy tất cả cells từ trên xuống
   - Kết quả: 6 files `.npz` + 3 artifacts

2. **`2_Federated_Training.ipynb`** (30-60 phút)
   - Chạy tất cả cells
   - Đợi training hoàn thành
   - Kết quả: `global_model.h5`

3. **`3_Model_Evaluation_Export.ipynb`** (5 phút)
   - Chạy tất cả cells
   - Kết quả: 3 PNG plots + metrics

---

## ✅ KIỂM TRA KẾT QUẢ

```bash
# Check files
ls -lh Output/models/
ls -lh Output/metrics/

# Xem plots
open Output/metrics/confusion_matrix.png
open Output/metrics/accuracy_plot.png
```

**Phải có:**
- ✅ `global_model.h5` (~10-20 MB)
- ✅ 3 PNG files (confusion matrix, accuracy, F1)
- ✅ `metrics_report.json`

---

## 🎯 SAU KHI TEST XONG

**Chạy Full Dataset:**

1. Sửa `training_config.yaml`:
   ```yaml
   experimental:
     use_sample_data: false  # ← Đổi thành false
   ```

2. Chạy lại 3 notebooks (5-7 giờ)

3. Kết quả: Model với accuracy > 95%

---

## ⚠️ LƯU Ý

- **Không tắt máy** khi đang training
- **Chạy qua đêm** cho full dataset
- **Backup** model sau khi train xong

---

## 🆘 GẶP LỖI?

| Lỗi | Giải pháp |
|-----|-----------|
| Out of Memory | Giảm `chunk_size: 10000` |
| Training chậm | Dùng 10% data trước |
| Kernel died | Restart kernel, chạy lại |

**Chi tiết:** Xem file `HUONG_DAN_TRAINING.md`

---

**Chúc bạn thành công! 🚀**
