import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Aktifkan autologging untuk Kriteria 2
mlflow.sklearn.autolog()

# 2. Load data bersih yang sudah Anda buat
# Pastikan path ini benar (sesuaikan jika file ada di folder lain)
df = pd.read_csv('namadataset_preprocessing/data_clean.csv')

# 3. Persiapan Model
X = df.drop(columns=['target']) # Pastikan kolom 'target' ada di dataset heart.csv Anda
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Training
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

print("Training selesai! Artefak telah dibuat otomatis.")
