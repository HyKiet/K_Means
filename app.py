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
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
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

    new_clusters_info = []
    for cid in range(model.n_clusters):
        cluster_df = df[df["Cluster"] == cid]
        count = len(cluster_df)
        if count > 0:
            avg_study = round(float(cluster_df["StudyHoursPerWeek"].mean()), 1)
            avg_abs = round(float(cluster_df["Absences"].mean()), 1)
            avg_gpa = round(float(cluster_df["GPA"].mean()), 2)
        else:
            avg_study = 0.0
            avg_abs = 0.0
            avg_gpa = 0.0
        
        info = labels_map[cid].copy()
        info["centroid"] = {
            "study_hours": avg_study,
            "absences": avg_abs,
            "gpa": avg_gpa
        }
        new_clusters_info.append({
            "id": cid,
            "count": count,
            **info
        })

    return jsonify({"success": True, "total": len(df),
                    "clusters": new_clusters_info,
                    "data": df.to_dict(orient="records")})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    API Tải lên file CSV chứa danh sách sinh viên.
    Phân tích bằng mô hình K-Means và chuẩn hóa dữ liệu, sau đó trả về danh sách gán nhãn
    kèm theo thống kê mới của các cụm dựa trên dữ liệu tải lên này.
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "Không tìm thấy tệp tin được tải lên"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Chưa chọn tệp tin"}), 400
    if not file.filename.endswith(".csv"):
        return jsonify({"success": False, "error": "Vui lòng tải lên tệp tin định dạng .csv"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"success": False, "error": f"Lỗi đọc file CSV: {str(e)}"}), 400

    # Kiểm tra các cột bắt buộc
    required = ["StudyHoursPerWeek", "Absences", "GPA"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return jsonify({"success": False, "error": f"Tệp CSV thiếu các cột bắt buộc: {', '.join(missing)}"}), 400

    # 1. Điền giá trị trống cho các cột học tập
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        mean_val = df[col].mean()
        if pd.isna(mean_val):
            mean_val = 0.0
        df[col] = df[col].fillna(mean_val)

    # 2. Loại bỏ bản ghi trùng lặp theo StudentID
    if "StudentID" in df.columns:
        df["StudentID"] = df["StudentID"].fillna("").astype(str)
        df = df.drop_duplicates(subset="StudentID", keep="first")
    else:
        df["StudentID"] = [f"SV{i+1:04d}" for i in range(len(df))]

    # Cập nhật các dòng StudentID bị trống
    df["StudentID"] = df["StudentID"].astype(str)
    for i in df.index:
        if df.at[i, "StudentID"].strip() == "":
            df.at[i, "StudentID"] = f"SV{i+1:04d}"

    # 3. Loại bỏ nhiễu/outlier (nằm ngoài khoảng hợp lệ) giống hệt lúc train
    mask = (
        df["StudyHoursPerWeek"].between(0, 50)
        & df["Absences"].between(0, 30)
        & df["GPA"].between(0, 4.0)
    )
    df = df[mask].reset_index(drop=True)

    if len(df) == 0:
        return jsonify({"success": False, "error": "Tệp CSV không còn dữ liệu hợp lệ nào sau khi loại bỏ nhiễu (outliers)"}), 400

    # Lấy dữ liệu và thực hiện chuẩn hóa + dự đoán cụm
    X_new = df[required].values
    X_new_scaled = scaler.transform(X_new)
    clusters = model.predict(X_new_scaled)
    df["Cluster"] = clusters.tolist()
    df["Label"] = [labels_map[c]["label"] for c in clusters]
    df["Color"] = [labels_map[c]["color"] for c in clusters]

    # Tính toán lại thông số tâm cụm thực tế và số lượng sinh viên cho tệp dữ liệu mới
    new_clusters_info = []
    for cid in range(model.n_clusters):
        cluster_df = df[df["Cluster"] == cid]
        count = len(cluster_df)
        if count > 0:
            avg_study = round(float(cluster_df["StudyHoursPerWeek"].mean()), 1)
            avg_abs = round(float(cluster_df["Absences"].mean()), 1)
            avg_gpa = round(float(cluster_df["GPA"].mean()), 2)
        else:
            avg_study = 0.0
            avg_abs = 0.0
            avg_gpa = 0.0
        
        info = labels_map[cid].copy()
        info["centroid"] = {
            "study_hours": avg_study,
            "absences": avg_abs,
            "gpa": avg_gpa
        }
        new_clusters_info.append({
            "id": cid,
            "count": count,
            **info
        })

    return jsonify({
        "success": True,
        "total": len(df),
        "clusters": new_clusters_info,
        "data": df.to_dict(orient="records")
    })


@app.route("/api/download_template")
def api_download_template():
    """Tải xuống tệp CSV mẫu để người dùng chạy thử."""
    csv_content = "StudentID,StudyHoursPerWeek,Absences,GPA\n" \
                  "SV0001,35.0,2.0,3.8\n" \
                  "SV0002,12.0,15.0,1.8\n" \
                  "SV0003,28.0,4.0,3.2\n" \
                  "SV0004,8.0,22.0,1.2\n" \
                  "SV0005,42.0,1.0,3.95\n" \
                  "SV0006,20.0,8.0,2.65\n"
    from flask import Response
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=students_sample.csv"}
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Server: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
