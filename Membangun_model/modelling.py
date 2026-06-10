import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.datasets import load_breast_cancer

# --- KONFIGURASI AUTENTIKASI ---
# Dengan menggunakan dagshub.init, library ini akan otomatis mendeteksi 
# DAGSHUB_TOKEN dari environment variable dan melakukan login ke DagsHub.
# Ini jauh lebih stabil daripada melakukan set_tracking_uri secara manual.

try:
    import dagshub
    dagshub.init(
        repo_owner='heriwibowo-dev', 
        repo_name='Eksperimen_SML_HeriWibowo', 
        mlflow=True
    )
    print("DagsHub & MLflow terinisialisasi dengan sukses.")
except Exception as e:
    print(f"Gagal inisialisasi DagsHub: {e}")
    # Fallback: Tetap set secara manual jika dagshub gagal
    mlflow.set_tracking_uri("https://dagshub.com/heriwibowo-dev/Eksperimen_SML_HeriWibowo.mlflow")

# --- LOADING DATA ---
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- TRAINING & TUNING ---
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_Tuning_Advanced"):
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    tuner = RandomizedSearchCV(rf, param_dist, n_iter=5, cv=3, random_state=42)
    
    print("Memulai training...")
    tuner.fit(X_train, y_train)

    print(f"Parameter Terbaik: {tuner.best_params_}")
    print("Training sukses dan artefak telah di-log ke DagsHub!")
