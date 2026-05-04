import faiss
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from rag.embeddings import embedding_engine

class VectorStore:
    def __init__(self, data_path: str = None):
        self.index = None
        self.documents = []
        self.dimension = 384  # MiniLM-L6-v2 dimension
        
        if data_path:
            self.load_data(data_path)

    def load_data(self, data_path: str):
        path = Path(data_path)
        if not path.exists():
            print(f"[WARN] CRM data not found at {data_path}")
            return

        with open(path, 'r') as f:
            self.documents = json.load(f)
        
        if not self.documents:
            print("[WARN] CRM data is empty")
            return

        # Prepare texts for embedding
        texts = [doc["content"] for doc in self.documents]
        embeddings = embedding_engine.encode(texts)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        print(f"Vector Store initialized with {len(self.documents)} documents")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.index is None:
            return []
            
        query_vector = embedding_engine.encode([query])
        distances, indices = self.index.search(query_vector.astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["score"] = float(distances[0][i])
                results.append(doc)
        
        return results

# Default instance (will be initialized in pipeline or api)
vector_store = VectorStore()
