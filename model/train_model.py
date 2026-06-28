# ============================================================
# train_model.py - Huan luyen mo hinh K-Means phan loai sinh vien
# De tai 6: Phan loai sinh vien dua tren ket qua hoc tap
# ============================================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. LOAD DU LIEU
# ============================================================
print("=" * 60)
print("1. LOAD DU LIEU")
print("=" * 60)

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'students.csv')
df = pd.read_csv(data_path)

print(f"So luong ban ghi: {len(df)}")
print(f"Cac cot: {list(df.columns)}")
print(f"\nMau du lieu (5 dong dau):")
print(df.head())
print(f"\nThong ke mo ta:")
print(df.describe())

# ============================================================
# GIAI THICH Y NGHIA CAC COT DU LIEU
# ============================================================
print("\n" + "=" * 60)
print("Y NGHIA CAC COT DU LIEU")
print("=" * 60)
print("""
| Cot                | Y nghia                           | Vai tro       |
|--------------------|-----------------------------------|---------------|
| StudentID          | Ma sinh vien (dinh danh)          | ID (bo qua)   |
| StudyHoursPerWeek  | So gio hoc moi tuan (0-40)        | INPUT Feature  |
| Absences           | So buoi vang trong ky (0-25)      | INPUT Feature  |
| GPA                | Diem trung binh tich luy (0-4.0)  | INPUT Feature  |

OUTPUT: Cluster (cum) - duoc tao boi K-Means (unsupervised learning)
Nguon data: Tu tao dua tren mo hinh sinh vien thuc te
         (Tham khao: Kaggle - Student Performance Factors)
         https://www.kaggle.com/datasets/lainguyn123/student-performance-factors
""")

# ============================================================
# 2. TIEN XU LY DU LIEU (PREPROCESSING)
# ============================================================
print("=" * 60)
print("2. TIEN XU LY DU LIEU (PREPROCESSING)")
print("=" * 60)

# 2a. Kiem tra gia tri null (missing values)
print("\n--- 2a. Kiem tra gia tri NULL ---")
print(f"Gia tri null trong dataset:")
print(df.isnull().sum())
total_null = df.isnull().sum().sum()
print(f"\n=> Tong so gia tri null: {total_null}")

# Xu ly null: Dien gia tri trung binh (mean) cho cac cot so
if total_null > 0:
    print("=> Xu ly: Dien gia tri TRUNG BINH (mean) cho cac gia tri null")
    for col in ['StudyHoursPerWeek', 'Absences', 'GPA']:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            print(f"   {col}: {null_count} null -> dien mean = {mean_val:.2f}")

print(f"\nSau xu ly null:")
print(df.isnull().sum())

# 2b. Kiem tra va xoa du lieu trung lap (duplicates)
print("\n--- 2b. Kiem tra du lieu TRUNG LAP ---")
dup_count = df.duplicated(subset='StudentID').sum()
print(f"So ban ghi trung lap (theo StudentID): {dup_count}")

if dup_count > 0:
    print(f"=> Xu ly: Xoa {dup_count} ban ghi trung lap, giu ban ghi dau tien")
    df = df.drop_duplicates(subset='StudentID', keep='first')
    print(f"   So ban ghi sau khi xoa: {len(df)}")

# 2c. Kiem tra va xu ly outlier (gia tri bat thuong)
print("\n--- 2c. Kiem tra OUTLIER (gia tri bat thuong) ---")
print(f"Kiem tra range hop le:")
print(f"  StudyHoursPerWeek: min={df['StudyHoursPerWeek'].min()}, max={df['StudyHoursPerWeek'].max()}")
print(f"  Absences: min={df['Absences'].min()}, max={df['Absences'].max()}")
print(f"  GPA: min={df['GPA'].min()}, max={df['GPA'].max()}")

# Loai bo outlier
outlier_mask = (
    (df['StudyHoursPerWeek'] < 0) | (df['StudyHoursPerWeek'] > 50) |
    (df['Absences'] < 0) | (df['Absences'] > 30) |
    (df['GPA'] < 0) | (df['GPA'] > 4.0)
)
outlier_count = outlier_mask.sum()
print(f"\nSo outlier tim thay: {outlier_count}")

if outlier_count > 0:
    print(f"=> Xu ly: Loai bo {outlier_count} ban ghi outlier")
    outlier_rows = df[outlier_mask]
    for _, row in outlier_rows.iterrows():
        print(f"   {row['StudentID']}: Study={row['StudyHoursPerWeek']}, "
              f"Abs={row['Absences']}, GPA={row['GPA']}")
    df = df[~outlier_mask]
    print(f"   So ban ghi sau khi loai: {len(df)}")

# 2d. Ket qua sau tien xu ly
print(f"\n--- KET QUA SAU TIEN XU LY ---")
print(f"So ban ghi cuoi cung: {len(df)}")
print(f"Null con lai: {df.isnull().sum().sum()}")
print(f"Trung lap con lai: {df.duplicated(subset='StudentID').sum()}")

cleaned_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'students_cleaned.csv')
df.to_csv(cleaned_data_path, index=False)
print(f"Da luu du lieu sach ra file: {cleaned_data_path}")

# ============================================================
# Chon features (bo cot StudentID vi no chi la dinh danh)
# ============================================================
features = ['StudyHoursPerWeek', 'Absences', 'GPA']
X = df[features].values

print(f"\nSo features: {len(features)}")
print(f"Features su dung: {features}")
print(f"Shape du lieu: {X.shape}")

# ============================================================
# 3. CHIA BO DU LIEU TRAIN VA TEST
# ============================================================
print("\n" + "=" * 60)
print("3. CHIA BO DU LIEU TRAIN VA TEST")
print("=" * 60)

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

print(f"Tong so du lieu: {len(X)} mau")
print(f"Ty le chia: 80% Train / 20% Test")
print(f"  - Du lieu Train: {len(X_train)} mau ({len(X_train)/len(X)*100:.0f}%)")
print(f"  - Du lieu Test:  {len(X_test)} mau ({len(X_test)/len(X)*100:.0f}%)")

# ============================================================
# 4. CHUAN HOA DU LIEU (StandardScaler)
# ============================================================
print("\n" + "=" * 60)
print("4. CHUAN HOA DU LIEU (StandardScaler)")
print("=" * 60)
print("""
K-Means su dung khoang cach Euclidean de tinh toan.
Neu cac features co scale khac nhau (VD: StudyHours 0-40, GPA 0-4),
feature co gia tri lon hon se THONG TRI ket qua.
=> Can chuan hoa de tat ca features co cung scale (mean=0, std=1).
""")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Truoc chuan hoa - Mean: {X_train.mean(axis=0).round(2)}")
print(f"Truoc chuan hoa - Std:  {X_train.std(axis=0).round(2)}")
print(f"Sau chuan hoa   - Mean: {X_train_scaled.mean(axis=0).round(4)}")
print(f"Sau chuan hoa   - Std:  {X_train_scaled.std(axis=0).round(4)}")

# ============================================================
# 5. XAC DINH SO CUM K TOI UU (Elbow Method)
# ============================================================
print("\n" + "=" * 60)
print("5. XAC DINH SO CUM K TOI UU (Elbow Method)")
print("=" * 60)

inertias = []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    km.fit(X_train_scaled)
    inertias.append(km.inertia_)
    print(f"  k={k}: Inertia = {km.inertia_:.2f}")

chart_dir = os.path.join(os.path.dirname(__file__), '..', 'static', 'images')
os.makedirs(chart_dir, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('So cum (k)', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.title('Elbow Method - Xac dinh so cum toi uu', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(list(K_range))
plt.tight_layout()
plt.savefig(os.path.join(chart_dir, 'elbow_chart.png'), dpi=150)
plt.close()
print(f"\n=> Bieu do Elbow da luu tai: static/images/elbow_chart.png")
print(f"=> Chon k=4 (diem khuy tay ro nhat)")

# ============================================================
# 6. HUAN LUYEN MO HINH K-MEANS
# ============================================================
print("\n" + "=" * 60)
print("6. HUAN LUYEN MO HINH K-MEANS")
print("=" * 60)

k = 4
model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10,
    max_iter=300
)

print("""
GIAI THICH CAC THAM SO CUA HAM KMeans():
-----------------------------------------
* n_clusters = 4:
  So cum (k) can phan chia. O day ta chia thanh 4 nhom:
  Xuat sac, Gioi, Trung binh, Yeu.

* random_state = 42:
  Hat giong (seed) cho bo sinh so ngau nhien.
  Dam bao ket qua GIONG NHAU moi lan chay (reproducible).

* n_init = 10:
  K-Means chay 10 lan voi cac tam cum ban dau khac nhau,
  sau do chon ket qua TOT NHAT (inertia thap nhat).

* max_iter = 300:
  So vong lap toi da cho MOI lan chay K-Means.
  Thuat toan se dung khi hoi tu HOAC dat 300 vong lap.
""")

print("GIAI THICH HAM fit():")
print("-" * 40)
print("""
model.fit(X_train_scaled):
  - Dau vao: X_train_scaled - ma tran du lieu da chuan hoa
  - Qua trinh:
    1. Khoi tao k=4 tam cum (centroid) ngau nhien
    2. Gan moi diem du lieu vao cum co centroid GAN NHAT
    3. Tinh lai centroid = trung binh cac diem trong cum
    4. Lap lai buoc 2-3 cho den khi HOI TU
  - Dau ra: Model da hoc duoc vi tri cac centroid
""")

model.fit(X_train_scaled)

print(f"Huan luyen thanh cong!")
print(f"   - So vong lap thuc te: {model.n_iter_}")
print(f"   - Inertia (tong khoang cach): {model.inertia_:.2f}")

# ============================================================
# 7. DANH GIA DO CHINH XAC
# ============================================================
print("\n" + "=" * 60)
print("7. DANH GIA DO CHINH XAC CUA THUAT TOAN")
print("=" * 60)

train_labels = model.predict(X_train_scaled)
test_labels = model.predict(X_test_scaled)

train_silhouette = silhouette_score(X_train_scaled, train_labels)
test_silhouette = silhouette_score(X_test_scaled, test_labels)

print(f"""
SILHOUETTE SCORE (Chi so danh gia chat luong phan cum):
-------------------------------------------------------
* Pham vi: -1 den 1
  - Gan 1:  Cac cum tach biet RO RANG (tot)
  - Gan 0:  Cac cum CHONG CHEO
  - Gan -1: Diem bi phan SAI cum

* Ket qua:
  - Train set: {train_silhouette:.4f}
  - Test set:  {test_silhouette:.4f}
""")

if train_silhouette > 0.5:
    print("NHAN XET: Silhouette Score > 0.5 => Mo hinh phan cum TOT!")
    print("   Cac cum tach biet ro rang, sinh vien duoc phan loai hop ly.")
elif train_silhouette > 0.25:
    print("NHAN XET: Silhouette Score 0.25-0.5 => Mo hinh phan cum KHA TOT.")
else:
    print("NHAN XET: Silhouette Score < 0.25 => Mo hinh can cai thien.")

print(f"\nSo luong sinh vien moi cum (tren Train set):")
for i in range(k):
    count = np.sum(train_labels == i)
    print(f"  Cum {i}: {count} sinh vien")

# ============================================================
# 8. XAC DINH Y NGHIA TUNG CUM
# ============================================================
print("\n" + "=" * 60)
print("8. XAC DINH Y NGHIA TUNG CUM")
print("=" * 60)

centroids_original = scaler.inverse_transform(model.cluster_centers_)

print(f"\nTam cum (Centroid) - gia tri goc:")
print(f"{'Cum':<6} {'StudyHours':<15} {'Absences':<12} {'GPA':<8}")
print("-" * 45)
for i, c in enumerate(centroids_original):
    print(f"  {i:<4} {c[0]:<15.1f} {c[1]:<12.1f} {c[2]:<8.2f}")

labels_map = {}
gpa_order = np.argsort(-centroids_original[:, 2])

label_names = ["Xuat sac", "Gioi", "Trung binh", "Yeu"]
label_descriptions = [
    "GPA cao (>3.5), hoc nhieu (>25h/tuan), it vang (<3 buoi)",
    "GPA kha (3.0-3.5), hoc tuong doi (18-25h/tuan), vang it (3-6 buoi)",
    "GPA trung binh (2.0-3.0), hoc vua phai (9-17h/tuan), vang nhieu (8-14 buoi)",
    "GPA thap (<2.0), it hoc (<8h/tuan), vang rat nhieu (>15 buoi)"
]
label_colors = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444"]

for rank, cluster_id in enumerate(gpa_order):
    labels_map[int(cluster_id)] = {
        "label": label_names[rank],
        "description": label_descriptions[rank],
        "color": label_colors[rank]
    }

print(f"\nY NGHIA CAC CUM:")
print("-" * 60)
for cluster_id in range(k):
    info = labels_map[cluster_id]
    print(f"  Cum {cluster_id} -> {info['label']}")
    print(f"    {info['description']}")
    print()

# ============================================================
# 9. VE BIEU DO PHAN CUM
# ============================================================
print("=" * 60)
print("9. VE BIEU DO PHAN CUM")
print("=" * 60)

X_all_scaled = scaler.transform(X)
all_labels = model.predict(X_all_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i in range(k):
    mask = all_labels == i
    axes[0].scatter(X[mask, 0], X[mask, 2],
                   c=labels_map[i]['color'], label=labels_map[i]['label'],
                   s=60, alpha=0.7, edgecolors='white', linewidth=0.5)
axes[0].set_xlabel('So gio hoc/tuan', fontsize=11)
axes[0].set_ylabel('GPA', fontsize=11)
axes[0].set_title('Phan cum: Gio hoc vs GPA', fontsize=13)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

for i in range(k):
    mask = all_labels == i
    axes[1].scatter(X[mask, 1], X[mask, 2],
                   c=labels_map[i]['color'], label=labels_map[i]['label'],
                   s=60, alpha=0.7, edgecolors='white', linewidth=0.5)
axes[1].set_xlabel('So buoi vang', fontsize=11)
axes[1].set_ylabel('GPA', fontsize=11)
axes[1].set_title('Phan cum: Vang mat vs GPA', fontsize=13)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(chart_dir, 'clusters_chart.png'), dpi=150)
plt.close()
print(f"=> Bieu do phan cum da luu tai: static/images/clusters_chart.png")

# ============================================================
# 10. XUAT MODEL RA FILE .pkl (DONG BANG MO HINH)
# ============================================================
print("\n" + "=" * 60)
print("10. XUAT MODEL RA FILE .pkl (DONG BANG MO HINH)")
print("=" * 60)

model_dir = os.path.dirname(__file__)

model_path = os.path.join(model_dir, 'kmeans_model.pkl')
joblib.dump(model, model_path)
print(f"Da luu model K-Means: {model_path}")

scaler_path = os.path.join(model_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"Da luu StandardScaler: {scaler_path}")

labels_path = os.path.join(model_dir, 'labels_map.pkl')
joblib.dump(labels_map, labels_path)
print(f"Da luu Labels Map: {labels_path}")

print("""
DONG BANG MO HINH (Model Serialization):
  * model.pkl: Mo hinh K-Means da huan luyen
  * scaler.pkl: Bo chuan hoa du lieu
  * labels_map.pkl: Ban do y nghia cac cum

  Su dung joblib.dump() de serialize (dong bang)
  va joblib.load() de deserialize (giai dong bang)
  khi tich hop vao ung dung web Flask.
""")

# ============================================================
# KIEM TRA: Load lai model va predict
# ============================================================
print("=" * 60)
print("KIEM TRA: Load model va du doan thu")
print("=" * 60)

loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)
loaded_labels = joblib.load(labels_path)

test_student = np.array([[25, 3, 3.5]])
test_scaled = loaded_scaler.transform(test_student)
prediction = loaded_model.predict(test_scaled)
cluster = int(prediction[0])

print(f"Input: StudyHours=25, Absences=3, GPA=3.5")
print(f"Cum du doan: {cluster}")
print(f"Phan loai: {loaded_labels[cluster]['label']}")
print(f"Mo ta: {loaded_labels[cluster]['description']}")
print(f"\n{'='*60}")
print(f"HOAN TAT HUAN LUYEN MO HINH!")
print(f"{'='*60}")
