from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import uvicorn

from schemas import ClassifyRequest, ClassifyResponse, StatusResponse, HistoryResponse, AnalyticsResponse
from services.predict_service import predict_service
from services.history_service import history_service
from inference.hybrid_predictor import _load_model

app = FastAPI(title="LeadSense AI API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup: Warm up the model
@app.on_event("startup")
async def startup_event():
    print("Warming up MiniLM model...")
    _load_model()
    from rag.rag_pipeline import rag_pipeline
    print("RAG System Initialized.")
    print("API Ready.")

# --- API Endpoints ---

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    return {"model_ready": True}


@app.post("/api/classify")
async def classify(request: ClassifyRequest):
    try:
        result = predict_service.classify_lead(
            text=request.text,
            lead_name=request.lead_name,
            company=request.company,
            source=request.source
        )
        
        # If input was invalid, return the specific error response
        if result.get("label") == "Invalid":
            return JSONResponse(status_code=400, content=result)
            
        history_service.add_record(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history", response_model=HistoryResponse)
async def get_history():
    return {"history": history_service.get_all()}

@app.get("/api/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    return {"session": history_service.get_analytics()}

@app.post("/api/session/clear")
async def clear_session():
    history_service.clear()
    return {"status": "success", "message": "Session cleared"}

# --- Static File Serving ---
# This allows the backend to serve the frontend directly
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/")
async def read_index():
    index_file = static_path / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "LeadSense AI API is running. UI not found in /static."}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
