import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.datasets import load_breast_cancer
import dagshub

# --- OTENTIKASI PAKSA ---
token = os.getenv('DAGSHUB_TOKEN')
user_name = os.getenv('DAGSHUB_USER_NAME') # Pastikan ini diset di main.yml

if not token or not user_name:
    raise EnvironmentError("Kredensial DagsHub tidak ditemukan di GitHub Secrets!")

# Login secara eksplisit
dagshub.login(token=token)

# Inisialisasi DagsHub untuk MLflow
dagshub.init(
    repo_owner=user_name, 
    repo_name='Eksperimen_SML_HeriWibowo', 
    mlflow=True
)

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
    rf = RandomForestClassifier(random_state=42)
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    tuner = RandomizedSearchCV(rf, param_dist, n_iter=5, cv=3, random_state=42)
    
    print("Memulai training...")
    tuner.fit(X_train, y_train)

    print(f"Parameter Terbaik: {tuner.best_params_}")
    print("Training sukses dan artefak telah di-log ke DagsHub!")
