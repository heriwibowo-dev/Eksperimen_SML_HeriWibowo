import os
import mlflow

# --- DETEKSI LINGKUNGAN ---
if os.getenv("GITHUB_ACTIONS") == "true":
    # 1. Pastikan MLflow tahu harus kemana
    mlflow.set_tracking_uri("https://dagshub.com/heriwibowo-dev/Eksperimen_SML_HeriWibowo.mlflow")
    
    # 2. Paksa MLflow menggunakan kredensial dari GitHub Secrets yang kita set di main.yml
    # MLflow akan membaca variabel ini secara otomatis untuk autentikasi HTTP
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')
else:
    import dagshub
    dagshub.init(repo_owner='heriwibowo-dev', repo_name='Eksperimen_SML_HeriWibowo', mlflow=True)

# ... lanjut ke kode training Anda ...

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
