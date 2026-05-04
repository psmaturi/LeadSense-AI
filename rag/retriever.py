from typing import List, Dict, Any
from rag.vector_store import vector_store

class Retriever:
    def __init__(self, top_k: int = 2):
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant context for a given query string."""
        return vector_store.search(query, top_k=self.top_k)

# Singleton instance
retriever = Retriever()
