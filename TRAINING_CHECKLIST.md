# TRAINING PROGRESS CHECKLIST

**Ngày bắt đầu:** _______________  
**Dự kiến hoàn thành:** _______________

---

## ✅ GIAI ĐOẠN 1: SETUP & CÀI ĐẶT

### 1.1 Kiểm Tra Hệ Thống
- [ ] Python 3.8+ đã cài đặt
- [ ] RAM >= 16GB
- [ ] Disk space >= 20GB trống
- [ ] Dataset (169 CSV files) có trong `DataSets/`

### 1.2 Cài Đặt Dependencies
- [ ] Chạy: `pip3 install tensorflow keras scikit-learn pandas numpy matplotlib seaborn pyyaml jupyter`
- [ ] Verify: `python3 -c "import tensorflow; print('OK')"`
- [ ] Check GPU (optional): `python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

### 1.3 Cấu Hình
- [ ] Mở `Notebooks/configs/training_config.yaml`
- [ ] Đặt `use_sample_data: true` (cho lần đầu)
- [ ] Đặt `sample_fraction: 0.1`

**Thời gian:** ~10 phút  
**Status:** ⬜ Chưa bắt đầu | ⏳ Đang làm | ✅ Hoàn thành

---

## ✅ GIAI ĐOẠN 2: DATA PREPROCESSING

### 2.1 Mở Jupyter Notebook
- [ ] Chạy: `cd Notebooks && jupyter notebook`
- [ ] Browser mở Jupyter interface

### 2.2 Chạy `1_Data_Preprocessing.ipynb`
- [ ] Cell 1-2: Setup and Imports → ✅ "All imports successful!"
- [ ] Cell 3: Load Configuration → ✅ Config loaded
- [ ] Cell 4: Load Dataset → ✅ "Dataset loaded successfully!"
  - **Thời gian:** _____ phút
  - **Số rows:** _______________
- [ ] Cell 5: Clean Data → ✅ "Data cleaned!"
- [ ] Cell 6: Encode Labels → ✅ "Labels encoded!"
  - **Files created:** `label_encoder.pkl`, `labels.json`
- [ ] Cell 7: Normalize Features → ✅ "Features normalized!"
  - **File created:** `scaler.pkl`
- [ ] Cell 8: Partition Data → ✅ "Data partitioned!"
  - **Client 0 samples:** _______________
  - **Client 1 samples:** _______________
  - **Client 2 samples:** _______________
  - **Client 3 samples:** _______________
  - **Client 4 samples:** _______________
  - **Test samples:** _______________
- [ ] Cell 9: Save Data → ✅ "All data saved successfully!"
- [ ] Cell 10: Verification → ✅ All files exist

### 2.3 Verify Outputs
```bash
ls -lh Output/data/      # Phải có 6 files .npz
ls -lh Output/models/    # Phải có 3 files (scaler, encoder, labels)
```
- [ ] `client_0_data.npz` exists
- [ ] `client_1_data.npz` exists
- [ ] `client_2_data.npz` exists
- [ ] `client_3_data.npz` exists
- [ ] `client_4_data.npz` exists
- [ ] `test_data.npz` exists
- [ ] `scaler.pkl` exists
- [ ] `label_encoder.pkl` exists
- [ ] `labels.json` exists

**Thời gian thực tế:** _____ phút  
**Status:** ⬜ Chưa bắt đầu | ⏳ Đang làm | ✅ Hoàn thành

---

## ✅ GIAI ĐOẠN 3: FEDERATED LEARNING TRAINING

### 3.1 Chạy `2_Federated_Training.ipynb`
- [ ] Cell 1: Setup → ✅ TensorFlow loaded
  - **GPU available:** ⬜ Yes | ⬜ No
- [ ] Cell 2: Load Config → ✅ Config loaded
  - **Num rounds:** _____
  - **Num clients:** _____
- [ ] Cell 3: Load Data → ✅ All client data loaded
- [ ] Cell 4: Create Model → ✅ Model created
  - **Total parameters:** _______________
  - **Model size:** _____ MB
- [ ] Cell 5: Initialize Server & Clients → ✅ Initialized

### 3.2 Main Training Loop (Cell 6)
**Bắt đầu:** ___:___ (giờ:phút)

- [ ] Round 1/30 → Accuracy: _____ %
- [ ] Round 5/30 → Accuracy: _____ %
- [ ] Round 10/30 → Accuracy: _____ %
- [ ] Round 15/30 → Accuracy: _____ %
- [ ] Round 20/30 → Accuracy: _____ %
- [ ] Round 25/30 → Accuracy: _____ %
- [ ] Round 30/30 → Accuracy: _____ %

**Kết thúc:** ___:___ (giờ:phút)  
**Tổng thời gian:** _____ giờ _____ phút

**Final Metrics:**
- **Test Accuracy:** _____ % (Mục tiêu: >95%)
- **Test Loss:** _____

### 3.3 Save Model
- [ ] Cell 8: Model saved → ✅ `global_model.h5` created
  - **Model size:** _____ MB
- [ ] Cell 9: History saved → ✅ `training_history.json` created
- [ ] Cell 10: Quick evaluation → ✅ Model verified

**Thời gian thực tế:** _____ giờ _____ phút  
**Status:** ⬜ Chưa bắt đầu | ⏳ Đang làm | ✅ Hoàn thành

---

## ✅ GIAI ĐOẠN 4: MODEL EVALUATION & EXPORT

### 4.1 Chạy `3_Model_Evaluation_Export.ipynb`
- [ ] Cell 1-3: Load Model & Data → ✅ Loaded
- [ ] Cell 4: Generate Predictions → ✅ Predictions generated
- [ ] Cell 5: Calculate Metrics → ✅ Metrics calculated
  - **Overall Accuracy:** _____ %
  - **Macro F1-Score:** _____
  - **Weighted F1-Score:** _____
- [ ] Cell 6: Per-Class Metrics → ✅ Table displayed
  - **Classes with F1 >= 0.85:** _____ / 34
  - **Classes with F1 < 0.85:** _____ / 34
- [ ] Cell 7-8: Confusion Matrix → ✅ Plot created & saved
- [ ] Cell 9: Training Curves → ✅ Plot created & saved
- [ ] Cell 10: F1-Score Chart → ✅ Plot created & saved
- [ ] Cell 11: Export Metrics → ✅ `metrics_report.json` created
- [ ] Cell 12: Classification Report → ✅ `classification_report.txt` created

### 4.2 Verify Outputs
```bash
ls -lh Output/metrics/
```
- [ ] `training_history.json` exists
- [ ] `metrics_report.json` exists
- [ ] `classification_report.txt` exists
- [ ] `confusion_matrix.png` exists
- [ ] `accuracy_plot.png` exists
- [ ] `f1_scores_per_class.png` exists

**Thời gian thực tế:** _____ phút  
**Status:** ⬜ Chưa bắt đầu | ⏳ Đang làm | ✅ Hoàn thành

---

## ✅ GIAI ĐOẠN 5: KIỂM TRA CUỐI CÙNG

### 5.1 Verify All Deliverables

**Models (cho Web App):**
- [ ] `Output/models/global_model.h5` (_____ MB)
- [ ] `Output/models/scaler.pkl`
- [ ] `Output/models/label_encoder.pkl`
- [ ] `Output/models/labels.json`

**Metrics (cho báo cáo):**
- [ ] `Output/metrics/training_history.json`
- [ ] `Output/metrics/metrics_report.json`
- [ ] `Output/metrics/classification_report.txt`

**Visualizations (cho thesis):**
- [ ] `Output/metrics/confusion_matrix.png` (300 DPI)
- [ ] `Output/metrics/accuracy_plot.png` (300 DPI)
- [ ] `Output/metrics/f1_scores_per_class.png` (300 DPI)

### 5.2 Quality Check
- [ ] Overall accuracy >= 95%
- [ ] All 34 classes have F1-Score >= 0.85
- [ ] Confusion matrix shows good diagonal dominance
- [ ] Training curves show convergence (accuracy tăng, loss giảm)
- [ ] Model file loadable: `python3 -c "from tensorflow import keras; keras.models.load_model('Output/models/global_model.h5'); print('OK')"`

### 5.3 Backup
- [ ] Copy `Output/` folder to backup location
- [ ] Backup command: `cp -r Output/ Output_backup_$(date +%Y%m%d)/`

**Status:** ⬜ Chưa bắt đầu | ⏳ Đang làm | ✅ Hoàn thành

---

## 📊 TỔNG KẾT

### Thời Gian Thực Tế

| Giai đoạn | Dự kiến | Thực tế | Ghi chú |
|-----------|---------|---------|---------|
| Setup | 10 mins | _____ | |
| Data Preprocessing | 5-10 mins | _____ | |
| FL Training | 30-60 mins | _____ | |
| Evaluation | 5 mins | _____ | |
| **TOTAL** | **40-75 mins** | **_____** | |

### Kết Quả Cuối Cùng

**Model Performance:**
- Overall Accuracy: _____ % (Target: >95%)
- Macro F1-Score: _____ (Target: >0.85)
- Classes meeting F1 threshold: _____ / 34

**Đạt Mục Tiêu:**
- [ ] ✅ Accuracy > 95%
- [ ] ✅ All classes F1 > 0.85
- [ ] ✅ All deliverables created
- [ ] ✅ Visualizations high quality

### Vấn Đề Gặp Phải & Giải Pháp

1. **Vấn đề:** _______________________________________________
   **Giải pháp:** _______________________________________________

2. **Vấn đề:** _______________________________________________
   **Giải pháp:** _______________________________________________

3. **Vấn đề:** _______________________________________________
   **Giải pháp:** _______________________________________________

---

## 🎯 NEXT STEPS

- [ ] Review tất cả visualizations
- [ ] Đưa plots vào báo cáo đồ án
- [ ] Chuẩn bị demo cho giảng viên
- [ ] Bắt đầu xây dựng Web App
- [ ] Viết phần kết quả trong thesis

---

## 📝 GHI CHÚ

_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________

---

**Hoàn thành:** ⬜ Chưa | ⏳ Đang làm | ✅ Xong  
**Ngày hoàn thành:** _______________  
**Người thực hiện:** Nguyễn Đức Thắng
