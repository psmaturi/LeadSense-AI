import uuid
import datetime
from typing import Dict
from inference.hybrid_predictor import predict_full
from rag.rag_pipeline import rag_pipeline

import re

def is_valid_input(text: str) -> bool:
    # 1. Reject empty or whitespace input
    text = text.strip()
    if not text:
        return False
    
    # 2. Reject very short input (<5 characters)
    if len(text) < 5:
        return False
    
    # 3. Reject inputs with only numbers (including negative/decimals)
    # This regex handles: 123, -123, 45.6, -45.6
    if re.fullmatch(r"^-?\d+(\.\d+)?$", text):
        return False
    
    # 4. Reject inputs with only special characters
    if re.fullmatch(r"[^a-zA-Z0-9\s]+", text):
        return False
    
    # 5. Reject inputs with less than 2 meaningful words
    # Meaningful word = contains at least one letter or number
    words = [w for w in text.split() if re.search(r"[a-zA-Z0-9]", w)]
    if len(words) < 2:
        return False
        
    return True

class PredictService:
    def classify_lead(self, text: str, lead_name: str = None, company: str = None, source: str = "email") -> Dict:
        # 0. Validation Layer
        if not is_valid_input(text):
            return {
                "label": "Invalid",
                "message": "Input has no meaningful intent"
            }

        # 1. RAG Layer: Retrieve Intelligence
        try:
            # We search based on lead text or name/company if available
            search_query = f"{text} {company or ''}"
            rag_result = rag_pipeline.run(search_query)
            retrieved_text = rag_result["full_context"]
            summary = rag_result["summary"]
            rag_enabled = True
        except Exception as e:
            print(f"[WARN] RAG Pipeline failed: {e}")
            retrieved_text = None
            summary = None
            rag_enabled = False

        # 2. Hybrid Classification Layer
        # Pass original text separately so rules don't scan RAG context
        if retrieved_text:
            context_aware_text = f"Input: {text}\n[HISTORICAL CRM CONTEXT]\n{retrieved_text}"
        else:
            context_aware_text = text
            
        prediction = predict_full(context_aware_text, raw_text=text)
        
        # 3. Format the response
        return {
            "record_id": str(uuid.uuid4()),
            "label": prediction["label"],
            "confidence": round(prediction["confidence"], 4),
            "probabilities": {k: round(v, 4) for k, v in prediction["probabilities"].items()},
            "engine": prediction["method"],
            "applied_rules": [],
            "lead_name": lead_name,
            "company": company,
            "source": source,
            "retrieved_context": retrieved_text,
            "intelligence_summary": summary,
            "rag_enabled": rag_enabled,
            "timestamp": datetime.datetime.now().isoformat()
        }

# Singleton instance
predict_service = PredictService()
