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

# 1. Inisialisasi DagsHub (Menggunakan kredensial dari environment)
dagshub.init(repo_owner='heriwibowo-dev', repo_name='Eksperimen_SML_HeriWibowo', mlflow=True)

# 2. Setup Data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# 3. Aktifkan Autologging dengan konfigurasi khusus untuk estimator.html
# Autologging ini akan otomatis men-generate: MLmodel, conda.yaml, model.pkl, 
# python_env.yaml, requirements.txt, dan estimator.html
mlflow.sklearn.autolog(log_models=True, log_input_examples=True, log_model_signatures=True)

with mlflow.start_run(run_name="RandomForest_Tuning_Advanced"):
    # 4. Training dengan Hyperparameter Tuning
    rf = RandomForestClassifier(random_state=42)
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    }
    tuner = RandomizedSearchCV(rf, param_dist, n_iter=3, cv=3)
    tuner.fit(X_train, y_train)

    # 5. Logging Manual untuk metric_info.json
    best_params = tuner.best_params_
    best_score = tuner.best_score_
    metric_info = {
        "best_params": best_params,
        "best_cv_score": best_score
    }
    with open("metric_info.json", "w") as f:
        json.dump(metric_info, f)
    mlflow.log_artifact("metric_info.json")

    # 6. Logging Confusion Matrix (Manual Artifact)
    y_pred = tuner.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names).plot(ax=ax)
    plt.title("Confusion Matrix")
    
    # Simpan plot sebagai artefak
    plt.savefig("training_confusion_matrix.png")
    mlflow.log_artifact("training_confusion_matrix.png")

    print("Training selesai. Artefak lengkap telah di-log ke DagsHub.")
