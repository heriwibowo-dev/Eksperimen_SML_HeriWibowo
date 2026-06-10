import os
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import dagshub

# 1. Konfigurasi Autentikasi yang Aman
token = os.getenv('DAGSHUB_TOKEN')
if not token:
    raise EnvironmentError("DAGSHUB_TOKEN tidak ditemukan di GitHub Secrets!")

dagshub.auth.add_app_token(token)
dagshub.init(repo_owner='heriwibowo-dev', repo_name='Eksperimen_SML_HeriWibowo', mlflow=True)

# 2. Setup Data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# 3. Aktifkan Autologging (Otomatis membuat: MLmodel, conda.yaml, model.pkl, python_env.yaml, requirements.txt, estimator.html)
mlflow.sklearn.autolog(log_models=True, log_input_examples=True, log_model_signatures=True)

with mlflow.start_run(run_name="RandomForest_Tuning_Advanced"):
    # 4. Training
    rf = RandomForestClassifier(random_state=42)
    param_dist = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
    tuner = RandomizedSearchCV(rf, param_dist, n_iter=3, cv=3)
    tuner.fit(X_train, y_train)

    # 5. Logging Manual: metric_info.json
    metric_info = {
        "best_params": tuner.best_params_,
        "best_cv_score": tuner.best_score_
    }
    with open("metric_info.json", "w") as f:
        json.dump(metric_info, f)
    mlflow.log_artifact("metric_info.json")

    # 6. Logging Manual: training_confusion_matrix.png
    y_pred = tuner.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names).plot(ax=ax)
    plt.title("Confusion Matrix")
    plt.savefig("training_confusion_matrix.png")
    mlflow.log_artifact("training_confusion_matrix.png")

    print("Training sukses. Semua artefak telah di-log.")
