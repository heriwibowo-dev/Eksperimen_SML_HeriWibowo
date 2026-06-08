import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Inisialisasi koneksi ke DagsHub
# Ini akan mengarahkan semua log MLflow ke repositori DagsHub Anda
dagshub.init(repo_owner='heriwibowo-dev', repo_name='Eksperimen_SML_HeriWibowo', mlflow=True)

# 2. Aktifkan autologging
# Ini akan menangkap metrik, parameter, dan model secara otomatis
mlflow.sklearn.autolog()

# 3. Load data bersih 
# Pastikan path ini sesuai dengan struktur folder Anda di Colab
df = pd.read_csv('/content/Eksperimen_SML_HeriWibowo/namadataset_preprocessing/data_clean.csv')

# 4. Persiapan Model
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Training
# Dengan mlflow.start_run(), semua hasil training akan tercatat di DagsHub
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

print("Training selesai! Artefak telah di-log ke DagsHub secara otomatis.")