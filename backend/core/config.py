import os

UPLOAD_DIR = "backend/data/uploads"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


os.makedirs(UPLOAD_DIR, exist_ok=True)

