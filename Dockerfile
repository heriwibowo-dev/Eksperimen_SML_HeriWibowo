# Menggunakan image dasar Python yang ringan
FROM python:3.12-slim

# Mengatur direktori kerja di dalam kontainer
WORKDIR /app

# Menyalin file requirements (pastikan Anda sudah punya file ini)
COPY requirements.txt .

# Menginstal dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh isi proyek ke dalam kontainer
COPY . .

# Menentukan perintah untuk menjalankan model atau aplikasi
# Sesuaikan jika Anda ingin menjalankan API atau skrip tertentu
CMD ["python", "Membangun_model/modelling.py"]
