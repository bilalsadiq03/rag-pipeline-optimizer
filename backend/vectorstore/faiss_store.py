import faiss
import numpy as np
from typing import List

class FAISSVectorStore:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.texts = []

    def add(self, embeddings: List[List[float]], texts: List[str]):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.texts.extend(texts)

    def search(self, query_embedding: List[float], top_k: int = 5):
        if self.index.ntotal == 0:
            return []

        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(
            query_vector,
            min(top_k, self.index.ntotal)
        )

        results = []
        for idx in indices[0]:
            if idx < len(self.texts):   
                results.append(self.texts[idx])

        return results
