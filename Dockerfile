# Menggunakan image dasar Python yang ringan
FROM python:3.12-slim

# Mengatur direktori kerja di dalam kontainer
WORKDIR /app

# Menginstal dependensi langsung (tanpa perlu file requirements.txt lagi)
RUN pip install --no-cache-dir mlflow docker pytest flake8 dagshub pandas scikit-learn joblib

# Menyalin seluruh isi proyek ke dalam kontainer
COPY . .

# Menentukan perintah untuk menjalankan model
# Pastikan path ini sesuai dengan struktur folder Anda
CMD ["python", "Membangun_model/modelling.py"]
