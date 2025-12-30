# Sửa Lỗi Chia Train/Test Dataset

## 🔍 Vấn Đề Phát Hiện

### 1. Tỷ lệ Train/Test không đúng
- **Mong đợi**: Train 80%, Test 20%
- **Thực tế**: Train 73.59%, Test 26.41%
- **Nguyên nhân**: Dữ liệu bị chia **2 lần**:
  - Lần 1 (Cell 6): Chia 95% train, 5% test → lưu vào CSV
  - Lần 2 (Cell 22 - `partition_data_noniid`): Lại chia thêm 80% train, 20% test
  - Kết quả: Actual test ratio = 26.41% thay vì 20%

### 2. Label 0 bị thiếu trong train set
- **Thực tế**: Label 0 có 0 samples trong train, 36 samples trong test
- **Nguyên nhân**: 
  - Label 0 chỉ có 36 samples trong toàn bộ dataset (rất ít)
  - Cell 6 dùng `test_size = 0.05` (5%) thay vì 0.2 (20%) theo config
  - Với 5% test size, các labels có quá ít samples không được stratify tốt
  - Một số labels chỉ rơi vào test set mà không có trong train set

### 3. Phân bổ labels không đầy đủ trong clients
- **Nguyên nhân**: Trong `partition_data_noniid()`, chỉ có 5 nhóm đầu tiên được phân bổ cho 5 clients
- Các labels còn lại (Web: 27-32, BruteForce: 33, Other: 0) không được xử lý đúng

---

## ✅ Giải Pháp Đã Áp Dụng

### 1. Sửa Cell 6: Dùng config thay vì hardcode
**Trước:**
```python
test_size = 0.05  # 5% for testing - HARDCODED!
random_seed = 42
```

**Sau:**
```python
# Use test split ratio from config (20% for test, 80% for train)
test_size = config['data']['test_split_ratio']  # From config: 0.2 (20%)
random_seed = config['random_seed']  # Use same seed as config

print(f"📋 Split configuration:")
print(f"   Test size: {test_size*100:.0f}%")
print(f"   Train size: {(1-test_size)*100:.0f}%")
```

### 2. Loại bỏ việc chia train/test lần 2 trong `partition_data_noniid()`
**Cách cũ:**
- Nhận `df_train` → chia lại thành train/test → partition cho clients
- Kết quả: Dữ liệu bị chia 2 lần!

**Cách mới:**
- Nhận `df_train` (đã được chia sẵn từ Cell 6) → partition trực tiếp cho clients
- Test set được thêm vào riêng sau khi partition
- Kết quả: Chỉ chia 1 lần duy nhất ở Cell 6

### 3. Sửa hàm `partition_data_noniid()` để đảm bảo tất cả labels được phân bổ
**Cải tiến:**
- Định nghĩa lại attack groups để bao gồm **tất cả 34 labels**
- Distribute data theo 2 bước:
  1. Mỗi client nhận 70% của nhóm attack chính
  2. 30% còn lại được phân bổ đều cho tất cả clients
- Verify tất cả samples đều được assign (không mất data)

### 4. Xử lý test set riêng biệt
**Các thay đổi trong notebook:**

**Cell 11**: Clean cả train và test
```python
df_train_clean = data_utils.clean_data(df_train)
df_test_clean = data_utils.clean_data(df_test)
```

**Cell 13**: GSA feature selection cho cả train và test
```python
df_train_filtered = data_utils.filter_features_by_names(df_train, selected_features, y_column)
df_test_filtered = data_utils.filter_features_by_names(df_test, selected_features, y_column)
```

**Cell 18**: Encode labels cho cả train và test
```python
df_train, label_encoder, label_mapping = data_utils.encode_labels(df_train, ...)
df_test[y_column] = label_encoder.transform(df_test[y_column])  # Use same encoder
```

**Cell 20**: Normalize features cho cả train và test
```python
df_train, scaler = data_utils.normalize_features(df_train, ...)
df_test[X_columns] = scaler.transform(df_test[X_columns])  # Use same scaler
```

**Cell 22**: Partition chỉ cho train, thêm test riêng
```python
client_data = data_utils.partition_data_noniid(df_train, num_clients=5, ...)
# Add test set separately
X_test = df_test.drop(columns=[y_column]).values
y_test = df_test[y_column].values
client_data['test'] = {'X': X_test, 'y': y_test}
```

---

## 🎯 Kết Quả Mong Đợi

Sau khi rerun notebook với những thay đổi trên:

### ✅ Tỷ lệ Train/Test đúng
- Train: ~80% (từ 2,487,431 × 0.8 ≈ 1,989,945 samples)
- Test: ~20% (từ 2,487,431 × 0.2 ≈ 497,486 samples)
- Actual test ratio: ~20.0% (thay vì 26.41%)

### ✅ Tất cả 34 labels có mặt trong cả train và test
- Với test_size = 0.2 (20%) thay vì 0.05 (5%), stratified split sẽ hoạt động tốt hơn
- Các labels hiếm (như Label 0) sẽ có samples trong cả 2 sets

### ✅ Không mất data
- Tổng samples trong clients = 100% train data
- Không có samples bị bỏ sót

---

## 📋 Các Bước Tiếp Theo

1. **Xóa cache cũ** (✅ Đã thực hiện):
   ```bash
   rm -f Output/preprocessed/*.csv
   rm -rf Output/data/*.npz
   ```

2. **Rerun notebook**: Chạy lại từ Cell 6 trở đi để:
   - Tạo lại train_dataset.csv và test_dataset.csv với tỷ lệ 80/20
   - Partition data với logic mới
   - Lưu .npz files mới

3. **Verify kết quả**: Kiểm tra Cell 25 (Verification cell) để confirm:
   - Actual test ratio ≈ 20.0%
   - Tất cả 34 labels present trong cả train và test
   - Stratification check passed

---

## 🧪 Test Results

Đã test với 100k samples và logic hoạt động đúng:
```
✅ All 100,000 training samples assigned successfully
✅ All 34 labels are covered in attack groups
📊 Train/Test ratio: 90.9% / 9.1% (với sample data)
```

---

## 📝 Files Đã Sửa

1. **`Notebooks/utils/data_utils.py`**:
   - Hàm `partition_data_noniid()`: Loại bỏ train/test split, cải thiện label allocation

2. **`Notebooks/1_Data_Preprocessing.ipynb`**:
   - Cell 6: Dùng config thay vì hardcode test_size
   - Cell 11: Clean cả train và test
   - Cell 13: GSA cho cả train và test
   - Cell 18: Encode labels cho cả train và test
   - Cell 20: Normalize features cho cả train và test
   - Cell 22: Partition + verify split ratio

---

## ⚠️ Lưu Ý

- **Thời gian xử lý**: Rerun Cell 6 sẽ mất thời gian vì phải load toàn bộ 169 CSV files
- **Config quan trọng**: Đảm bảo `configs/training_config.yaml` có `test_split_ratio: 0.2`
- **Stratified split**: Với test_size = 0.2, sklearn sẽ stratify tốt hơn cho các labels hiếm

---

Được sửa bởi: AI Assistant  
Ngày: 2025-01-30




