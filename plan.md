# KẾ HOẠCH TRIỂN KHAI ĐỒ ÁN: ỨNG DỤNG FEDERATED LEARNING PHÁT HIỆN TẤN CÔNG MẠNG IOT

**Mục tiêu giai đoạn này:** Xây dựng, huấn luyện và đánh giá model Federated Learning (FL) để đạt độ chính xác cao nhất, sau đó xuất ra các file model để phục vụ việc tích hợp vào Web App cảnh báo sau này.

---

## PHẦN 1: CHUẨN BỊ DỮ LIỆU & MÔI TRƯỜNG (Setup)

### 1. Chọn Dataset (Bộ dữ liệu)
Vì là phát hiện tấn công IoT, bạn cần chọn dataset chuẩn.
* **Gợi ý:** N-BaIoT, CICIDS2017, hoặc UNSW-NB15.
* **Yêu cầu:** Dataset phải có nhãn (Label) rõ ràng (Ví dụ: `Normal`, `DDoS`, `Mirai`, `Scan`...).

### 2. Cấu trúc File Notebook (`.ipynb`)
Nên chia thành 3 file Notebook chính để dễ quản lý và debug:
1.  `1_Data_Preprocessing.ipynb`: Xử lý dữ liệu.
2.  `2_Federated_Training.ipynb`: Huấn luyện theo cơ chế FL.
3.  `3_Model_Evaluation_Export.ipynb`: Đánh giá và Xuất file.

---

## PHẦN 2: CHI TIẾT TRIỂN KHAI CODE (Execution Plan)

### BƯỚC 1: Tiền xử lý dữ liệu (`1_Data_Preprocessing.ipynb`)
Nhiệm vụ: Biến dữ liệu thô thành dữ liệu sạch mà Model hiểu được, và chia nhỏ dữ liệu cho các Client (IoT Devices giả lập).

1.  **Load Data:** Đọc file csv (pandas).
2.  **Cleaning:** Xử lý các giá trị Null/NaN, loại bỏ các cột không cần thiết (IP nguồn, IP đích có thể gây overfitting, nên tập trung vào packet size, protocol...).
3.  **Label Encoding:** Chuyển đổi nhãn tấn công từ chữ sang số.
    * *Lưu ý quan trọng:* Phải lưu lại mapping này (Ví dụ: 0: Normal, 1: DDoS) để sau này hiển thị lên Web.
4.  **Normalization (Chuẩn hóa):** Sử dụng `MinMaxScaler` hoặc `StandardScaler`.
    * *Cực kỳ quan trọng:* Phải lưu file `scaler.pkl` sau bước này. Web App sau này cần file này để chuẩn hóa traffic mới trước khi đưa vào model dự đoán.
5.  **Data Partitioning (Chia dữ liệu cho Client):**
    * Chia dữ liệu thành các phần (Client 1, Client 2, Client 3...).
    * Giả lập môi trường thực tế: Có thể chia theo kiểu IID (ngẫu nhiên đều) hoặc Non-IID (mỗi client có đặc thù dữ liệu tấn công khác nhau).
6.  **Save Data:** Lưu các tập dữ liệu đã xử lý (`client1_data.npy`, `client2_data.npy`, `test_data.npy`) để dùng cho bước sau.

### BƯỚC 2: Huấn luyện Federated Learning (`2_Federated_Training.ipynb`)
Nhiệm vụ: Giả lập Server và Clients để train model mà không gộp dữ liệu.

1.  **Define Model Architecture:**
    * Xây dựng mạng Neural Network (DNN, CNN 1D hoặc LSTM) phù hợp với input shape của dữ liệu mạng.
    * Input: Số lượng features (đặc trưng gói tin).
    * Output: Số lượng lớp (Softmax layer = số loại tấn công).
2.  **Define Client Logic:**
    * Hàm `client_update(model, local_data)`: Client tải model global về, train trên dữ liệu cục bộ của nó, tính toán trọng số (weights) mới.
3.  **Define Server Logic (Aggregation):**
    * Sử dụng thuật toán **FedAvg** (Federated Averaging): Server nhận weights từ các Client, tính trung bình cộng để cập nhật Global Model.
4.  **Training Loop (Vòng lặp huấn luyện):**
    * Thực hiện khoảng 10-50 vòng (Rounds).
    * Mỗi vòng: Server gửi model -> Client train -> Client gửi lại -> Server gộp.
    * Theo dõi Loss và Accuracy qua từng vòng.

### BƯỚC 3: Đánh giá & Xuất Model (`3_Model_Evaluation_Export.ipynb`)
Nhiệm vụ: Kiểm chứng độ chính xác và đóng gói sản phẩm.

1.  **Load Global Model:** Lấy model cuối cùng sau khi train xong.
2.  **Evaluate on Test Set:** Chạy model trên tập dữ liệu kiểm thử (Test set) chưa từng được train.
3.  **Visualization (Trực quan hóa):**
    * Vẽ biểu đồ Accuracy & Loss qua các vòng lặp.
    * **Confusion Matrix (Ma trận nhầm lẫn):** Rất quan trọng để biết model có nhầm lẫn giữa "Bình thường" và "Tấn công" hay không, hoặc nhầm giữa các loại tấn công nào.
    * Tính các chỉ số: Precision, Recall, F1-Score (Ưu tiên F1-Score vì dữ liệu tấn công thường mất cân bằng).
4.  **Export (Xuất file cho Web App):**
    * Lưu model: `global_model_iot_attack.h5` (nếu dùng Keras/TensorFlow) hoặc `.pth` (nếu dùng PyTorch).
    * Lưu Labels class: `label_classes.json` (để Web hiển thị tên tấn công).

---

## PHẦN 3: KẾT QUẢ ĐẦU RA (Deliverables)

Sau khi hoàn thành giai đoạn này, bạn cần có trong tay folder `resources` chứa:
1.  `global_model.h5`: Bộ não AI đã học xong.
2.  `scaler.pkl`: Công cụ chuẩn hóa dữ liệu đầu vào.
3.  `labels.json`: File định nghĩa (Ví dụ: `{0: "Normal", 1: "UDP Flood", 2: "TCP Flood"}`).
4.  `metrics_report.png`: Ảnh chụp biểu đồ độ chính xác (để báo cáo đồ án).

---

## GHI CHÚ CHO GIAI ĐOẠN SAU (WEB/APP)
Để chuẩn bị cho việc làm Web cảnh báo sau này, hãy nhớ:
* Hệ thống Web sẽ load file `global_model.h5` lên.
* Khi có traffic mạng đi qua, Web dùng `scaler.pkl` để xử lý, sau đó đưa vào Model.
* Nếu Model trả về kết quả khác "Normal" -> Kích hoạt gửi Mail/SMS.