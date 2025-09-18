from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class Embedder:
    def __init__(self, texts):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(texts)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        self.texts = texts

    def retrieve(self, query, k=3, return_distances=True):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        if return_distances:
            return [(self.texts[i], float(distances[0][j])) for j, i in enumerate(indices[0])]
        else:
            return [self.texts[i] for i in indices[0]]
