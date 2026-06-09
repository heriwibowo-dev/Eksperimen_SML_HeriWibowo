import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Inisialisasi koneksi ke DagsHub (Hanya jalan jika BUKAN di GitHub Actions)
if os.getenv('GITHUB_ACTIONS') != 'true':
    import dagshub
    dagshub.init(repo_owner='heriwibowo-dev', repo_name='Eksperimen_SML_HeriWibowo', mlflow=True)
    print("DagsHub terinisialisasi.")
else:
    print("Berjalan di GitHub Actions: Lewati inisialisasi DagsHub interaktif.")

# 2. Aktifkan autologging
mlflow.sklearn.autolog()

# 3. Load data bersih
# PENTING: Di GitHub Actions, path /content/... tidak ada. 
# Gunakan path relatif dari root proyek Anda.
# Jika data ada di folder 'namadataset_preprocessing', gunakan:
df = pd.read_csv('namadataset_preprocessing/data_clean.csv')

# 4. Persiapan Model
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Training
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

print("Training selesai! Artefak telah di-log.")
