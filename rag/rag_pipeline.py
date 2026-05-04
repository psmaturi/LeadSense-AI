from typing import Dict, Any, List
from rag.retriever import retriever
from rag.context_builder import context_builder
from rag.vector_store import vector_store
from pathlib import Path

class RAGPipeline:
    def __init__(self):
        # Auto-initialize vector store if data exists
        data_path = Path(__file__).resolve().parent.parent / "data" / "crm_history.json"
        if data_path.exists():
            vector_store.load_data(str(data_path))

    def run(self, query: str) -> Dict[str, Any]:
        """Execute the full RAG flow."""
        # 1. Retrieve
        relevant_docs = retriever.retrieve(query)
        
        # 2. Build Context
        full_context = context_builder.format_context(relevant_docs)
        summary = context_builder.get_summary(relevant_docs)
        
        return {
            "retrieved_docs": relevant_docs,
            "full_context": full_context,
            "summary": summary,
            "source_count": len(relevant_docs)
        }

# Singleton instance
rag_pipeline = RAGPipeline()
