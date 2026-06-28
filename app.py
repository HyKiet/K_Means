# ============================================================
# app.py — Flask Backend cho ứng dụng phân loại sinh viên K-Means
# ============================================================
# KIẾN TRÚC:
#   Frontend (HTML/CSS/JS) ←→ Flask API ←→ Model .pkl (Scikit-learn)
#   KHÔNG sử dụng Node.js — chỉ dùng Python Flask
# ============================================================

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

# ============================================================
# KHỞI TẠO FLASK APP
# ============================================================
app = Flask(__name__)
CORS(app)  # Cho phép Cross-Origin requests

# ============================================================
# LOAD MODEL ĐÃ ĐÓNG BĂNG (.pkl)
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Load 3 file .pkl đã được serialize bằng joblib
model = joblib.load(os.path.join(MODEL_DIR, 'kmeans_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
labels_map = joblib.load(os.path.join(MODEL_DIR, 'labels_map.pkl'))

print("✅ Đã load model K-Means từ file .pkl thành công!")
print(f"   - kmeans_model.pkl: Mô hình K-Means (k={model.n_clusters})")
print(f"   - scaler.pkl: StandardScaler")
print(f"   - labels_map.pkl: Bản đồ ý nghĩa các cụm")

# ============================================================
# API ROUTES (ENDPOINTS)
# ============================================================

# ----------------------------------------------------------
# ROUTE 1: GET / — Trang chủ
# ----------------------------------------------------------
# Method: GET
# Mô tả: Render trang HTML chính (index.html)
# Response: HTML page
# ----------------------------------------------------------
@app.route('/')
def index():
    """Render trang chủ web app."""
    return render_template('index.html')


# ----------------------------------------------------------
# ROUTE 2: POST /api/predict — Dự đoán phân loại sinh viên
# ----------------------------------------------------------
# Method: POST
# Content-Type: application/json
# Body: { "study_hours": 25, "absences": 3, "gpa": 3.5 }
# Response: {
#   "success": true,
#   "cluster": 0,
#   "label": "Xuất sắc",
#   "description": "GPA cao, học nhiều, ít vắng",
#   "color": "#10b981"
# }
# ----------------------------------------------------------
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API dự đoán phân loại sinh viên.
    
    Flow:
    1. Nhận dữ liệu JSON từ frontend
    2. Validate input
    3. Chuẩn hóa dữ liệu bằng scaler.pkl
    4. Dự đoán cụm bằng kmeans_model.pkl
    5. Tra cứu ý nghĩa cụm từ labels_map.pkl
    6. Trả kết quả JSON
    """
    try:
        data = request.get_json()
        
        # Validate input
        study_hours = float(data.get('study_hours', 0))
        absences = float(data.get('absences', 0))
        gpa = float(data.get('gpa', 0))
        
        # Kiểm tra giá trị hợp lệ
        if not (0 <= study_hours <= 50):
            return jsonify({"success": False, "error": "Số giờ học phải từ 0-50"}), 400
        if not (0 <= absences <= 30):
            return jsonify({"success": False, "error": "Số buổi vắng phải từ 0-30"}), 400
        if not (0 <= gpa <= 4.0):
            return jsonify({"success": False, "error": "GPA phải từ 0-4.0"}), 400
        
        # Tạo input array
        input_data = np.array([[study_hours, absences, gpa]])
        
        # Chuẩn hóa bằng scaler đã fit
        input_scaled = scaler.transform(input_data)
        
        # Dự đoán cụm
        cluster = int(model.predict(input_scaled)[0])
        
        # Tra cứu ý nghĩa
        cluster_info = labels_map[cluster]
        
        return jsonify({
            "success": True,
            "cluster": cluster,
            "label": cluster_info["label"],
            "description": cluster_info["description"],
            "color": cluster_info["color"]
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ----------------------------------------------------------
# ROUTE 3: GET /api/clusters — Thông tin tất cả các cụm
# ----------------------------------------------------------
# Method: GET
# Response: { "clusters": [...], "centroids": [...] }
# ----------------------------------------------------------
@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    """
    API trả thông tin tất cả các cụm.
    Bao gồm: tên, mô tả, màu sắc, và tâm cụm (centroid).
    """
    try:
        # Chuyển centroid về scale gốc
        centroids_original = scaler.inverse_transform(model.cluster_centers_)
        
        clusters = []
        for i in range(model.n_clusters):
            info = labels_map[i]
            centroid = centroids_original[i]
            clusters.append({
                "id": i,
                "label": info["label"],
                "description": info["description"],
                "color": info["color"],
                "centroid": {
                    "study_hours": round(float(centroid[0]), 1),
                    "absences": round(float(centroid[1]), 1),
                    "gpa": round(float(centroid[2]), 2)
                }
            })
        
        return jsonify({"success": True, "clusters": clusters})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ----------------------------------------------------------
# ROUTE 4: GET /api/dataset — Dữ liệu gốc + phân cụm
# ----------------------------------------------------------
# Method: GET
# Response: { "data": [...], "total": 50 }
# ----------------------------------------------------------
@app.route('/api/dataset', methods=['GET'])
def get_dataset():
    """
    API trả toàn bộ dataset sinh viên kèm kết quả phân cụm.
    Dùng để hiển thị bảng dữ liệu trên frontend.
    """
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, 'students_cleaned.csv'))
        
        # Predict cluster cho tất cả sinh viên
        features = df[['StudyHoursPerWeek', 'Absences', 'GPA']].values
        features_scaled = scaler.transform(features)
        clusters = model.predict(features_scaled)
        
        # Thêm cột cluster và label
        df['Cluster'] = clusters
        df['Label'] = [labels_map[c]['label'] for c in clusters]
        df['Color'] = [labels_map[c]['color'] for c in clusters]
        
        return jsonify({
            "success": True,
            "data": df.to_dict(orient='records'),
            "total": len(df)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# CHẠY SERVER
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🚀 Flask Server đang chạy!")
    print("   URL: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
