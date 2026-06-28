# ============================================================
# app.py - Flask Backend (Machine Learning Service)
# ============================================================
# Kien truc:  Frontend (HTML/CSS/JS)  <->  Flask API  <->  Model .pkl
# KHONG dung Node.js - chi dung Python Flask.
#
# 4 endpoint:
#   GET  /              -> trang dashboard
#   GET  /api/clusters  -> thong tin 4 cum + tam cum
#   GET  /api/dataset   -> toan bo du lieu kem nhan cum
#   POST /api/predict   -> phan loai 1 sinh vien moi
# ============================================================

import os
import joblib       # Dùng để đọc/ghi các file model đã được huấn luyện (.pkl)
import numpy as np  # Xử lý mảng và ma trận tính toán số học
import pandas as pd # Xử lý dữ liệu dạng bảng (DataFrame) giống như Excel
from flask import Flask, render_template, request, jsonify # Thư viện tạo web server Backend
from flask_cors import CORS # Cho phép Frontend (nếu khác domain) gọi được API của Flask

app = Flask(__name__) # Khởi tạo ứng dụng web Flask
CORS(app)             # Bật tính năng CORS chống chặn request chéo domain

BASE = os.path.dirname(__file__)
FEATURES = ["StudyHoursPerWeek", "Absences", "GPA"]

# --- Load mô hình đã đóng băng (.pkl) ---
# Tại sao phải đóng băng? Vì quá trình train rất mất thời gian. Ta chỉ train 1 lần, 
# lưu ra file .pkl, khi chạy web chỉ việc load lên để dùng luôn.
model = joblib.load(os.path.join(BASE, "model", "kmeans_model.pkl")) # Model K-Means (chứa thông tin 4 cụm)
scaler = joblib.load(os.path.join(BASE, "model", "scaler.pkl"))      # Bộ chuẩn hóa dữ liệu
labels_map = joblib.load(os.path.join(BASE, "model", "labels_map.pkl")) # Map id cụm ra ý nghĩa (Tên, màu sắc, nhận xét)
print(f"[OK] Đã load model K-Means (k={model.n_clusters})")


@app.route("/")
def index():
    """Trang dashboard chinh."""
    return render_template("index.html")


@app.route("/api/clusters")
def api_clusters():
    """Tra ve thong tin 4 cum: ten, mo ta, mau, icon, tam cum (centroid)."""
    clusters = []
    for cid in range(model.n_clusters):
        info = labels_map[cid]
        clusters.append({"id": cid, **info})
    return jsonify({"success": True, "clusters": clusters})


@app.route("/api/dataset")
def api_dataset():
    """Tra toan bo du lieu sinh vien (da lam sach) kem nhan cum tu mo hinh."""
    df = pd.read_csv(os.path.join(BASE, "data", "students_cleaned.csv"))
    labels = model.predict(scaler.transform(df[FEATURES].values))
    df["Cluster"] = labels
    df["Label"] = [labels_map[c]["label"] for c in labels]
    df["Color"] = [labels_map[c]["color"] for c in labels]
    return jsonify({"success": True, "total": len(df),
                    "data": df.to_dict(orient="records")})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    API Phân loại 1 sinh viên mới (khi người dùng bấm nút trên Web).
    Body JSON nhận vào: {"study_hours": 25, "absences": 3, "gpa": 3.5}
    Quy trình (Flow): Nhận dữ liệu -> Kiểm tra tính hợp lệ -> Chuẩn hóa (scaler) -> Dự đoán (model.predict) -> Trả về kết quả.
    """
    data = request.get_json(silent=True) or {}
    try:
        sh = float(data.get("study_hours"))
        ab = float(data.get("absences"))
        gpa = float(data.get("gpa"))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Du lieu khong hop le"}), 400

    # Kiểm tra khoảng giá trị có hợp lý không (validate business logic)
    checks = [(sh, 0, 50, "Giờ học"), (ab, 0, 30, "Số buổi vắng"), (gpa, 0, 4.0, "GPA")]
    for val, lo, hi, name in checks:
        if not (lo <= val <= hi):
            # Nếu giá trị lọt ra ngoài vùng cho phép -> trả về mã lỗi 400 (Bad Request)
            return jsonify({"success": False, "error": f"{name} phải nằm trong khoảng {lo}-{hi}"}), 400

    # Bước 1: scaler.transform() -> Đưa dữ liệu người dùng nhập về cùng tỷ lệ (chuẩn hóa) với tập data lúc train
    # Bước 2: model.predict() -> Thuật toán K-Means tính khoảng cách đến 4 tâm cụm và chọn cụm gần nhất
    cluster = int(model.predict(scaler.transform(np.array([[sh, ab, gpa]])))[0])
    
    # Bước 3: Lấy toàn bộ thông tin giải thích (tên nhóm, màu sắc, ý nghĩa) dựa vào id của cụm
    info = labels_map[cluster]
    
    # Bước 4: Trả về Frontend định dạng JSON
    return jsonify({"success": True, "cluster": cluster,
                    "input": {"study_hours": sh, "absences": ab, "gpa": gpa}, **info})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Server: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
