import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor # Contoh algoritma
import joblib

# 1. Mulai eksperimen MLflow
mlflow.set_experiment("Eksperimen_Model_ML")

with mlflow.start_run():
    # --- Kode training Anda di sini ---
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    # 2. Log parameter
    mlflow.log_param("n_estimators", 100)
    
    # 3. Log metrik (opsional tapi bagus untuk submission)
    score = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", score)
    
    # 4. Log model (menyimpan artefak)
    mlflow.sklearn.log_model(model, "model")
    
    print("Model dan parameter berhasil di-log ke MLflow!")
