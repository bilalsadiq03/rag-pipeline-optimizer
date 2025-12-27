import uuid
import os
from fastapi import UploadFile

def save_upload_file(upload_dir: str, file: UploadFile) -> str:
    file_ext = file.filename.split('.')[-1]
    file_name = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(upload_dir, file_name)

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    return file_path