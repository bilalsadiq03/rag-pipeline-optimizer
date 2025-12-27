from fastapi import FastAPI, UploadFile, File
from core.config import UPLOAD_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from utils.file_utils import save_upload_file
from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text


app = FastAPI(title="RAG Pipeline Optimizer")

@app.get("/")
def health_check():
    return {"status": "Backend is running"}

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    file_path = save_upload_file(UPLOAD_DIR, file)

    text = load_pdf(file_path)
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    
    return {
        "file_name": file.filename,
        "num_chunks": len(chunks),
        "chunks": chunks[0][:300]
    }