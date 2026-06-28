# K-Means Student Clustering Project

## 1. Giới thiệu Dự án
Dự án này là một ứng dụng web mô phỏng việc phân loại (clustering) sinh viên dựa trên kết quả học tập và thói quen của họ bằng thuật toán học máy K-Means.

Dự án sử dụng mô hình học máy đã được huấn luyện trước bằng **Scikit-learn** và đóng gói thành các file `.pkl`. Sau đó, một server **Flask** được xây dựng để cung cấp các API kết nối với giao diện web Frontend (HTML/CSS/JS).

## 2. Yêu cầu Đề bài (Requirements)
Từ kiến trúc và tính năng của hệ thống, chúng ta có thể thấy các yêu cầu chính của đề bài bao gồm:

1. **Phân cụm dữ liệu**: Áp dụng thuật toán K-Means để nhóm các sinh viên thành các cụm (cluster) có đặc điểm học tập tương đồng nhau. Dữ liệu đầu vào gồm 3 yếu tố:
   - `study_hours` (Số giờ học/tuần, từ 0 đến 50)
   - `absences` (Số buổi vắng, từ 0 đến 30)
   - `gpa` (Điểm trung bình học tập, từ 0 đến 4.0)
2. **Backend**:
   - Xây dựng một ứng dụng web không sử dụng Node.js, thay vào đó dùng **Python Flask**.
   - Load các mô hình `.pkl` (`kmeans_model.pkl`, `scaler.pkl`, `labels_map.pkl`) vào Flask backend.
   - Cung cấp API dự đoán phân loại cho một sinh viên mới: `POST /api/predict`.
   - Cung cấp API hiển thị thông tin các cụm đã phân: `GET /api/clusters`.
   - Cung cấp API xem toàn bộ tập dữ liệu gốc kèm theo nhãn: `GET /api/dataset`.
3. **Frontend**:
   - Sử dụng giao diện cơ bản (HTML/CSS/JS) gọi trực tiếp đến API của Flask.
   - Hiển thị trang chủ `/` với giao diện tương tác và thân thiện.
4. **Luồng dữ liệu (Data Flow)**:
   - Frontend gửi dữ liệu (JSON) -> Flask API -> Validate -> Tiền xử lý (Scale) -> Dự đoán với model K-Means -> Map kết quả với nhãn tương ứng -> Trả về JSON hiển thị cho người dùng.

## 3. Cấu trúc Thư mục
- `app.py`: File code chính khởi chạy Flask server và cấu hình các endpoints API.
- `model/`: Chứa các model học máy đã đóng băng.
  - `kmeans_model.pkl`: Model phân cụm.
  - `scaler.pkl`: StandardScaler dùng để chuẩn hóa input.
  - `labels_map.pkl`: Cấu hình ý nghĩa của các cụm phân loại.
- `data/`: Chứa các dữ liệu đầu vào.
  - `students_cleaned.csv`: Tập dữ liệu chuẩn hóa của sinh viên.
- `static/`: Chứa các file tĩnh css, js (nếu có).
- `templates/`: Chứa trang `index.html`.
- `requirements.txt`: Các thư viện Python cần thiết.

## 4. Hướng dẫn Chạy Ứng dụng
1. Cài đặt các thư viện yêu cầu:
   ```bash
   pip install -r requirements.txt
   ```
2. Chạy server Flask:
   ```bash
   python app.py
   ```
3. Truy cập vào giao diện web qua trình duyệt:
   ```
   http://localhost:5000
   ```
