import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# 1. Konfigurasi
# Ganti FOLDER_ID dengan ID folder di link Google Drive Anda
# Contoh: Jika linknya drive.google.com/drive/folders/1abc123... maka ID-nya adalah 1abc123...
FOLDER_ID = 'MASUKKAN_FOLDER_ID_ANDA_DISINI'
FILE_PATH = 'model_hasil_training.pkl'  # Sesuaikan dengan nama file model Anda

def upload_file():
    # 2. Setup Kredensial dari file yang dibuat di GitHub Action
    creds = service_account.Credentials.from_service_account_file(
        'gcp_key.json', 
        scopes=['https://www.googleapis.com/auth/drive']
    )
    service = build('drive', 'v3', credentials=creds)

    # 3. Upload file
    file_metadata = {
        'name': os.path.basename(FILE_PATH),
        'parents': [FOLDER_ID]
    }
    media = MediaFileUpload(FILE_PATH, mimetype='application/octet-stream')
    
    file = service.files().create(
        body=file_metadata, 
        media_body=media, 
        fields='id'
    ).execute()
    
    print(f"File berhasil di-upload dengan ID: {file.get('id')}")

if __name__ == '__main__':
    if os.path.exists(FILE_PATH):
        upload_file()
    else:
        print(f"Error: {FILE_PATH} tidak ditemukan!")
