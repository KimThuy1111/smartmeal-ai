from google.cloud import firestore

# Khởi tạo client Firestore (sử dụng Application Default Credentials hoặc file service account)
db = firestore.Client()