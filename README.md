# Phân loại sinh viên bằng K-Means — Đề tài 6

Dự án này là một ứng dụng web **Dashboard** mô phỏng việc phân loại (clustering) sinh viên dựa trên kết quả học tập bằng thuật toán **K-Means** (Machine Learning - Scikit-learn).

## 1. Yêu Cầu Của Đề Tài (Checklist)
- **Thiết kế Website:** Giao diện trực quan bằng HTML/CSS/JS thuần, hiển thị 4 KPI, biểu đồ phân tán (Chart.js) và bảng dữ liệu.
- **Backend (Không Node.js):** Sử dụng 100% **Python Flask** làm Machine Learning Service.
- **Mô Hình Đóng Băng (.pkl):** Thuật toán được train sẵn và serialize thành file `model.pkl` (Scikit-learn).
- **Dataset (Kaggle):** Tham khảo từ [Kaggle – Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors).
  - INPUT: `StudyHoursPerWeek` (0-50), `Absences` (0-30), `GPA` (0-4.0).
  - OUTPUT: Cluster sinh viên (Phân nhóm học lực).

## 2. Cấu trúc thư mục (Dành cho AI & Lập trình viên)
```text
K_Means/
├── app.py                       # File chính: Khởi động Flask Server (Port 5000)
├── requirements.txt             # Chứa danh sách các thư viện Python cần cài đặt
├── Dockerfile                   # Script để build Docker Image (đóng gói dự án)
├── data/
│   ├── students.csv             # Dữ liệu gốc
│   └── students_cleaned.csv     # Dữ liệu đã qua tiền xử lý (xóa null, trùng, outlier)
├── model/
│   ├── train_model.py           # Code huấn luyện và tiền xử lý dữ liệu từ đầu
│   ├── kmeans_model.pkl         # K-Means model (K=4)
│   ├── scaler.pkl               # StandardScaler
│   └── labels_map.pkl           # Config ý nghĩa các cụm
├── templates/index.html         # Giao diện frontend
└── static/css|js|images/        # Static resources (Chart.js, style...)
```

## 3. Hướng Dẫn Cài Đặt Và Chạy (Local - Môi trường gốc)

Nếu bạn không dùng Docker và muốn chạy trực tiếp bằng Python trên máy tính:

**Yêu cầu hệ thống:**
- Cài đặt Python (phiên bản 3.10 trở lên). *Lưu ý: Bắt buộc phải tích chọn "Add python.exe to PATH" khi cài đặt.*

**Các bước thực hiện:**
1. Mở Terminal (Command Prompt / PowerShell) trỏ vào thư mục gốc của dự án `K_Means`.
2. Cài đặt toàn bộ thư viện bắt buộc:
   ```bash
   pip install -r requirements.txt
   ```
3. Chạy Server Flask:
   ```bash
   python app.py
   ```
4. Truy cập trình duyệt ở đường dẫn: 👉 `http://localhost:5000`

---

## 4. Hướng Dẫn Chạy Bằng DOCKER 

Sử dụng Docker đảm bảo môi trường dự án chạy đúng 100% ở bất kỳ thiết bị nào mà không cần cài đặt Python. 

**Yêu cầu:** Đã cài đặt và mở **Docker Desktop** (Engine running).

**Các bước thực hiện (Nhập trên Terminal tại thư mục dự án):**

1. Build Image (Chờ tải thư viện Python vào bộ nhớ Docker):
   ```bash
   docker build -t kmeans_app .
   ```
2. Khởi chạy Container (Mở port 8080):
   ```bash
   docker run -d -p 8080:5000 --name kmeans_container kmeans_app
   ```
3. Truy cập trang web qua Docker bằng đường link: 👉 `http://localhost:8080`

*(Ghi chú: Để giải quyết lỗi font chữ hoặc hiển thị Emoji Unicode trong console Docker, file Dockerfile đã thêm `ENV PYTHONIOENCODING=utf-8`)*

---

## 5. API Documentation

Backend Flask cung cấp các API để thao tác với dữ liệu (cấu trúc JSON):

| Method | Endpoint | Chức năng |
|---|---|---|
| GET | `/` | Render giao diện HTML chính. |
| GET | `/api/clusters` | Trả về thông tin ý nghĩa 4 cụm + tâm cụm (centroid). |
| GET | `/api/dataset` | Lấy toàn bộ danh sách dữ liệu sinh viên kèm nhãn cụm. |
| POST | `/api/predict` | API dự đoán cụm cho một học sinh mới đưa vào. |

**Ví dụ Gọi `POST /api/predict`:**
```json
// Request (JSON)
{
  "study_hours": 35,
  "absences": 1,
  "gpa": 3.9
}

// Response (JSON)
{
  "success": true,
  "cluster": 0,
  "label": "Xuat sac",
  "description": "GPA cao (>3.5), hoc nhieu (>25h/tuan), it vang (<3 buoi)",
  "color": "#10b981"
}
```
