import os
import dagshub
import mlflow

# --- KONFIGURASI AUTENTIKASI ---
# Pastikan nama variabel di os.getenv() SAMA PERSIS dengan di main.yml
token = os.getenv('DAGSHUB_TOKEN')
user_name = os.getenv('DAGSHUB_USERNAME') # Kita samakan dengan nama secret di GitHub

if not token or not user_name:
    # Ini adalah pesan error yang muncul, mari kita perjelas pesan errornya
    raise EnvironmentError(f"Cek GitHub Secrets: token={bool(token)}, username={bool(user_name)}")

# Login secara eksplisit
dagshub.auth.add_app_token(token)

# Inisialisasi
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
