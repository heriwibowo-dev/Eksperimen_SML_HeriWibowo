import os
import dagshub
import mlflow

# --- KONFIGURASI OTOMATIS DAGSHUB ---
# Library dagshub secara otomatis mencari environment variable 
# bernama DAGSHUB_TOKEN dan DAGSHUB_USER_NAME (atau repo_owner)
token = os.getenv('DAGSHUB_TOKEN')
user_name = os.getenv('DAGSHUB_USERNAME')

if not token:
    raise EnvironmentError("DAGSHUB_TOKEN tidak ditemukan!")
if not user_name:
    # Sebagai backup, jika DAGSHUB_USERNAME tidak ada, gunakan hardcode ini
    user_name = 'heriwibowo-dev'

# Login dan Inisialisasi
dagshub.auth.add_app_token(token)
dagshub.init(
    repo_owner=user_name, 
    repo_name='Eksperimen_SML_HeriWibowo', 
    mlflow=True
)

# --- SISA KODE ANDA ---
# (Pastikan di bawah ini tidak ada lagi set_tracking_uri manual)
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
