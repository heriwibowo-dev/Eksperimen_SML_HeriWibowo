import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
# TAMBAHKAN BARIS INI:
from sklearn.datasets import load_breast_cancer 

# --- DETEKSI LINGKUNGAN ---
if os.getenv("GITHUB_ACTIONS") == "true":
    mlflow.set_tracking_uri("https://dagshub.com/heriwibowo-dev/Eksperimen_SML_HeriWibowo.mlflow")
else:
    import dagshub
    dagshub.init(repo_owner='heriwibowo-dev', repo_name='Eksperimen_SML_HeriWibowo', mlflow=True)

# Sekarang baris data = load_breast_cancer() akan berjalan lancar

# ... kode training Anda ...

# 2. Load data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Setup Tuning
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_Tuning_Advanced"):
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(random_state=42)
    tuner = RandomizedSearchCV(rf, param_dist, n_iter=5, cv=3, random_state=42)
    tuner.fit(X_train, y_train)

    print(f"Parameter Terbaik: {tuner.best_params_}")
    print("Training selesai dan artefak berhasil di-log ke DagsHub!")
