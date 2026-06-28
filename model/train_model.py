# ============================================================
# train_model.py - Huan luyen mo hinh K-Means phan loai sinh vien
# De tai 6: Phan loai sinh vien dua tren ket qua hoc tap
# ============================================================
# Quy trinh: Load -> Tien xu ly -> Chia train/test -> Chuan hoa
#            -> Chon K (Elbow) -> Train -> Danh gia -> Gan nhan
#            -> Ve bieu do -> Xuat .pkl
# ============================================================

import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")  # in duoc emoji tren Windows
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import joblib

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "..", "data")
IMG = os.path.join(BASE, "..", "static", "images")
os.makedirs(IMG, exist_ok=True)

FEATURES = ["StudyHoursPerWeek", "Absences", "GPA"]


def section(title):
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


# ============================================================
# 1. LOAD DU LIEU + GIAI THICH Y NGHIA CAC COT
# ============================================================
section("1. LOAD DU LIEU")
df = pd.read_csv(os.path.join(DATA, "students.csv"))
print(f"So ban ghi: {len(df)} | Cac cot: {list(df.columns)}")
print(df.head())

print("""
Y NGHIA CAC COT (Nguon tham khao: Kaggle - Student Performance Factors):
| Cot               | Y nghia                          | Vai tro       |
|-------------------|----------------------------------|---------------|
| StudentID         | Ma sinh vien (dinh danh)         | ID (bo qua)   |
| StudyHoursPerWeek | So gio tu hoc moi tuan (0-50)    | INPUT         |
| Absences          | So buoi vang trong ky (0-30)    | INPUT         |
| GPA               | Diem trung binh tich luy (0-4.0)| INPUT         |
INPUT  = 3 dac trung tren.
OUTPUT = Cluster (nhom sinh vien) do K-Means tu tim ra (hoc khong giam sat).
""")

# ============================================================
# 2. TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)
# ============================================================
section("2. TIEN XU LY DU LIEU")

# Bước 2a. Xử lý giá trị rỗng (NULL/NaN)
# Dùng hàm fillna() của pandas để điền giá trị Trung Bình (mean) vào những ô bị trống
print("Null truoc xu ly:\n", df[FEATURES].isnull().sum().to_string())
for col in FEATURES:
    df[col] = df[col].fillna(df[col].mean())

# 2b. Xoa ban ghi TRUNG LAP theo StudentID
before = len(df)
df = df.drop_duplicates(subset="StudentID", keep="first")
print(f"\nXoa trung lap: {before - len(df)} ban ghi")

# 2c. Loai bo OUTLIER (ngoai khoang hop le)
mask = (
    df["StudyHoursPerWeek"].between(0, 50)
    & df["Absences"].between(0, 30)
    & df["GPA"].between(0, 4.0)
)
print(f"Loai outlier: {(~mask).sum()} ban ghi")
df = df[mask].reset_index(drop=True)

print(f"=> Con lai {len(df)} ban ghi sach. Null: {df[FEATURES].isnull().sum().sum()}")
df.to_csv(os.path.join(DATA, "students_cleaned.csv"), index=False)

X = df[FEATURES].values

# ============================================================
# 3. CHIA TRAIN / TEST (80/20)
# ============================================================
section("3. CHIA BO DU LIEU TRAIN / TEST")
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
print(f"Tong: {len(X)} mau | Train: {len(X_train)} (80%) | Test: {len(X_test)} (20%)")

# ============================================================
# 4. CHUẨN HÓA DỮ LIỆU (StandardScaler)
# ============================================================
section("4. CHUAN HOA DU LIEU (StandardScaler)")
print("K-Means dung khoang cach Euclidean -> phai dua cac cot ve cung thang do.")
# Khởi tạo bộ chuẩn hóa. Thuật toán sẽ tính giá trị: Z = (X - Mean) / Std
scaler = StandardScaler()

# fit_transform: Vừa tìm Mean/Std trên tập Train, vừa chuẩn hóa tập Train
X_train_s = scaler.fit_transform(X_train)   
# transform: Dùng Mean/Std của tập Train áp dụng luôn cho tập Test (không tìm lại)
X_test_s = scaler.transform(X_test)

print(f"Sau chuan hoa - Mean ~ {X_train_s.mean(axis=0).round(2)}, Std ~ {X_train_s.std(axis=0).round(2)}")

# ============================================================
# 5. CHON SO CUM K TOI UU (Elbow Method)
# ============================================================
section("5. CHON SO CUM K TOI UU (Elbow + Silhouette)")
Ks = range(2, 9)
inertias, sils = [], []
for k in Ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_train_s)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_train_s, km.labels_))
    print(f"  k={k}: inertia={km.inertia_:7.1f} | silhouette={sils[-1]:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(list(Ks), inertias, "o-", color="#2f6f6a", lw=2)
ax[0].set(title="Elbow Method", xlabel="Số cụm (k)", ylabel="Inertia")
ax[0].axvline(4, color="#ef4444", ls="--", alpha=.7)
ax[0].grid(alpha=.3)
ax[1].plot(list(Ks), sils, "o-", color="#10b981", lw=2)
ax[1].set(title="Silhouette Score", xlabel="Số cụm (k)", ylabel="Score")
ax[1].axvline(4, color="#ef4444", ls="--", alpha=.7)
ax[1].grid(alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG, "elbow_chart.png"), dpi=130)
plt.close()
print("=> Chon k=4 (diem 'cui cho' ro, silhouette cao) -> 4 nhom: Xuat sac/Gioi/TB/Yeu")

# ============================================================
# 6. HUAN LUYEN MO HINH K-MEANS
# ============================================================
section("6. HUAN LUYEN MO HINH (giai thich tham so)")
print("""
KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
  n_clusters=4 : so nhom can chia.
  random_state=42 : co dinh seed -> ket qua lap lai duoc.
  n_init=10 : chay 10 lan voi tam khoi tao khac nhau, chon ket qua tot nhat.
  max_iter=300 : so vong lap toi da moi lan chay.

.fit(X_train_s) - thuat toan lap:
  1) Khoi tao 4 tam cum (centroid).
  2) Gan moi diem vao tam GAN NHAT (khoang cach Euclidean).
  3) Cap nhat lai tam = trung binh cac diem trong cum.
  4) Lap buoc 2-3 den khi hoi tu.
""")
# Khởi tạo mô hình K-Means với các tham số:
# - n_clusters=4: Chia làm 4 nhóm
# - random_state=42: Cố định seed khởi tạo ngẫu nhiên để lần chạy nào kết quả cũng giống nhau
# - n_init=10: Thuật toán sẽ chạy 10 lần với 10 bộ điểm trung tâm ngẫu nhiên khác nhau, sau đó chọn kết quả có Inertia thấp nhất
# - max_iter=300: Giới hạn tối đa 300 vòng lặp cho mỗi lần chạy (nếu chưa hội tụ)
model = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)

# Tiến hành chạy thuật toán phân cụm trên tập Train đã được chuẩn hóa
model.fit(X_train_s)
print(f"Hoi tu sau {model.n_iter_} vong lap | Inertia = {model.inertia_:.1f}")

# ============================================================
# 7. DANH GIA DO CHINH XAC
# ============================================================
section("7. DANH GIA (Silhouette Score)")
sil_train = silhouette_score(X_train_s, model.labels_)
sil_test = silhouette_score(X_test_s, model.predict(X_test_s))
print(f"Silhouette - Train: {sil_train:.3f} | Test: {sil_test:.3f}")
verdict = "TOT (cum tach biet ro)" if sil_train > 0.5 else (
    "KHA" if sil_train > 0.25 else "CAN CAI THIEN")
print(f"Nhan xet: Silhouette in [-1, 1], cang gan 1 cang tot => {verdict}")

# ============================================================
# 8. GAN Y NGHIA CHO TUNG CUM
# ============================================================
section("8. XAC DINH Y NGHIA TUNG CUM")
centers = scaler.inverse_transform(model.cluster_centers_)  # tam cum ve thang do goc
order = np.argsort(-centers[:, 2])  # sap xep theo GPA giam dan

names = ["Xuất sắc", "Giỏi", "Trung bình", "Yếu"]
descs = [
    "GPA cao, học nhiều, ít vắng - cần duy trì và giao nhiệm vụ thử thách.",
    "GPA khá, học ổn định - chỉ cần thêm động lực để bứt phá.",
    "GPA trung bình, vắng khá nhiều - cần nhắc nhở và hỗ trợ học tập.",
    "GPA thấp, ít học, vắng nhiều - cần cảnh báo và kèm sớm.",
]
colors = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444"]
icons = ["🌟", "👍", "📘", "⚠️"]

labels_map = {}
for rank, cid in enumerate(order):
    c = centers[cid]
    labels_map[int(cid)] = {
        "label": names[rank], "description": descs[rank],
        "color": colors[rank], "icon": icons[rank],
        "centroid": {"study_hours": round(float(c[0]), 1),
                     "absences": round(float(c[1]), 1),
                     "gpa": round(float(c[2]), 2)},
    }
for cid in range(4):
    m = labels_map[cid]
    print(f"  Cum {cid} -> {m['icon']} {m['label']:11} | "
          f"Study={m['centroid']['study_hours']}, Abs={m['centroid']['absences']}, GPA={m['centroid']['gpa']}")

# ============================================================
# 9. BIEU DO PHAN CUM
# ============================================================
section("9. VE BIEU DO PHAN CUM")
all_labels = model.predict(scaler.transform(X))
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for cid in range(4):
    m = all_labels == cid
    ax[0].scatter(X[m, 0], X[m, 2], c=labels_map[cid]["color"], label=labels_map[cid]["label"], s=35, alpha=.75)
    ax[1].scatter(X[m, 1], X[m, 2], c=labels_map[cid]["color"], label=labels_map[cid]["label"], s=35, alpha=.75)
ax[0].set(title="Giờ học và GPA", xlabel="Giờ học / tuần", ylabel="GPA")
ax[1].set(title="Vắng mặt và GPA", xlabel="Số buổi vắng", ylabel="GPA")
for a in ax:
    a.legend(fontsize=8)
    a.grid(alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG, "clusters_chart.png"), dpi=130)
plt.close()
print("=> Da luu clusters_chart.png")

# ============================================================
# 10. XUAT MODEL RA .pkl (dong bang mo hinh)
# ============================================================
section("10. XUAT FILE .pkl")
joblib.dump(model, os.path.join(BASE, "kmeans_model.pkl"))
joblib.dump(scaler, os.path.join(BASE, "scaler.pkl"))
joblib.dump(labels_map, os.path.join(BASE, "labels_map.pkl"))
print("Da luu: kmeans_model.pkl, scaler.pkl, labels_map.pkl")
print("\nHOAN TAT HUAN LUYEN!")
