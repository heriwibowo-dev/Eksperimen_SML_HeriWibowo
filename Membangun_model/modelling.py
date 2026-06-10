import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.datasets import load_breast_cancer

# --- KONFIGURASI AUTENTIKASI ---
# Kita mengambil variabel yang dikirim dari GitHub Actions (main.yml)
TRACKING_URI = "https://dagshub.com/heriwibowo-dev/Eksperimen_SML_HeriWibowo.mlflow"

if os.getenv("GITHUB_ACTIONS") == "true":
    # Pastikan username dan password tersedia di environment
    # Ini akan mengambil dari 'env' yang didefinisikan di main.yml
    mlflow.set_tracking_uri(TRACKING_URI)
    
    # Memastikan MLflow menggunakan variabel ini untuk autentikasi HTTP
    # Jika gagal, program akan berhenti dan memberi tahu kita lewat log
    if not os.getenv('MLFLOW_TRACKING_USERNAME') or not os.getenv('MLFLOW_TRACKING_PASSWORD'):
        raise EnvironmentError("Kredensial MLflow tidak ditemukan di GitHub Secrets!")
else:
    # Untuk penggunaan di lokal/Colab
    import dagshub
    dagshub.init(repo_owner='heriwibowo-dev', repo_name='Eksperimen_SML_HeriWibowo', mlflow=True)

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
    print("Training sukses dan artefak telah di-log ke DagsHub!")import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.datasets import load_breast_cancer

# --- KONFIGURASI AUTENTIKASI ---
# Kita mengambil variabel yang dikirim dari GitHub Actions (main.yml)
TRACKING_URI = "https://dagshub.com/heriwibowo-dev/Eksperimen_SML_HeriWibowo.mlflow"

if os.getenv("GITHUB_ACTIONS") == "true":
    # Pastikan username dan password tersedia di environment
    # Ini akan mengambil dari 'env' yang didefinisikan di main.yml
    mlflow.set_tracking_uri(TRACKING_URI)
    
    # Memastikan MLflow menggunakan variabel ini untuk autentikasi HTTP
    # Jika gagal, program akan berhenti dan memberi tahu kita lewat log
    if not os.getenv('MLFLOW_TRACKING_USERNAME') or not os.getenv('MLFLOW_TRACKING_PASSWORD'):
        raise EnvironmentError("Kredensial MLflow tidak ditemukan di GitHub Secrets!")
else:
    # Untuk penggunaan di lokal/Colab
    import dagshub
    dagshub.init(repo_owner='heriwibowo-dev', repo_name='Eksperimen_SML_HeriWibowo', mlflow=True)

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
