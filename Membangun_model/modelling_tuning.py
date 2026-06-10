import mlflow
import mlflow.sklearn
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# 1. Persiapan Data (Sesuaikan dengan dataset Anda)
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# 2. Setup Eksperimen
mlflow.set_experiment("Eksperimen_SML_NamaSiswa")

with mlflow.start_run():
    # Training Model
    n_estimators = 100
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    
    # Prediksi untuk metrik
    preds = model.predict(X_test)
    
    # Hitung Metrik
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average='weighted'),
        "recall": recall_score(y_test, preds, average='weighted'),
        "loss": 0.15, # Sesuaikan dengan perhitungan loss model Anda
        "n_estimators": n_estimators
    }

    # 3. MLflow Logging (Otomatis menghasilkan MLmodel, conda.yaml, dll)
    mlflow.log_params({"n_estimators": n_estimators})
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model") # Folder 'model/' akan tercipta otomatis

    # 4. Simpan metric_info.json (Untuk dibaca Grafana)
    with open("metric_info.json", "w") as f:
        json.dump(metrics, f)

    print("Training selesai. Artefak dan metric_info.json telah dibuat.")
