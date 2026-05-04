from typing import List, Dict
import datetime

class HistoryService:
    def __init__(self):
        self.records = []

    def add_record(self, record: Dict):
        self.records.insert(0, record)  # Newest first

    def get_all(self) -> List[Dict]:
        return self.records

    def clear(self):
        self.records = []

    def get_analytics(self) -> Dict:
        if not self.records:
            return {
                "Hot": 0, "Warm": 0, "Cold": 0,
                "trend": []
            }
        
        counts = {"Hot": 0, "Warm": 0, "Cold": 0}
        trend = []
        
        for r in self.records:
            label = r.get("label")
            if label in counts:
                counts[label] += 1
            trend.append({"confidence": r.get("confidence")})
        
        # Limit trend to last 10-20 points for UI
        return {
            **counts,
            "trend": list(reversed(trend[-20:]))
        }

# Singleton instance
history_service = HistoryService()
