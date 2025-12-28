from fastapi import FastAPI, UploadFile, File
from core.config import UPLOAD_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from utils.file_utils import save_upload_file
from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedder import LocalEmbedder
from vectorstore.faiss_store import FAISSVectorStore
from retrieval.retriever import Retriever


app = FastAPI(title="RAG Pipeline Optimizer")
embedder = LocalEmbedder()
vector_store = FAISSVectorStore(embedding_dim=384)
retriever = Retriever(embedder, vector_store)


@app.on_event("startup")
def startup_event():
    embedder = LocalEmbedder()
    vector_store = FAISSVectorStore(embedding_dim=384)

    retriever = Retriever(
        embedder=embedder,          
        vector_store=vector_store
    )

    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.retriever = retriever

@app.get("/")
def health_check():
    return {"status": "Backend is running"}


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    file_path = save_upload_file(UPLOAD_DIR, file)

    text = load_pdf(file_path)
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    
    embeddings = app.state.embedder.embed(chunks)
    app.state.vector_store.add(embeddings, chunks)

    return {
        "file_name": file.filename,
        "num_chunks": len(chunks),
        "chunks": chunks[0][:300]
    }


@app.post("/search")
def search(query: str):
    results = app.state.retriever.retrieve(query)

    return {
        "results": results,
        "faiss_vectors": app.state.vector_store.index.ntotal
    }