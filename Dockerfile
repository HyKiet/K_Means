# ============================================================
# Dockerfile - Dong goi ung dung Flask K-Means
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Cai thu vien truoc (tan dung cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toan bo source code
COPY . .

# Cong ung dung
EXPOSE 5000

# Fix encoding issue
ENV PYTHONIOENCODING=utf-8

# Chay ung dung
CMD ["python", "app.py"]
