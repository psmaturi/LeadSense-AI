from typing import List, Dict, Any

class ContextBuilder:
    def format_context(self, docs: List[Dict[str, Any]]) -> str:
        if not docs:
            return "No previous interaction history found for this lead."
        
        formatted_notes = []
        for doc in docs:
            note = f"[{doc.get('date', 'N/A')}] {doc.get('interaction_type', 'Note')}: {doc.get('content', '')}"
            formatted_notes.append(note)
            
        return "\n".join(formatted_notes)

    def get_summary(self, docs: List[Dict[str, Any]]) -> str:
        """Create a shorter summary for the UI."""
        if not docs:
            return "New customer - no history."
            
        # For now, just take the first one or combine
        return docs[0].get('content', '')[:150] + "..."

# Singleton instance
context_builder = ContextBuilder()
