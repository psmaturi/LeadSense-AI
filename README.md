# LeadSense AI — Intelligent Lead Priority Engine

LeadSense AI is a professional B2B lead classification system designed to prioritize incoming leads based on purchase intent. It combines a fine-tuned **MiniLM Transformer** model with a deterministic **Rule-Based Engine** and **RAG (Retrieval-Augmented Generation)** to provide highly accurate, context-aware lead scoring.

## 🚀 Features

*   **Hybrid Inference:** Combines neural classification (Transformer) with rule-based overrides for high-confidence predictions.
*   **RAG Pipeline:** Integrates historical CRM context using FAISS to improve classification accuracy for returning leads.
*   **Real-time Dashboard:** Modern, glassmorphism-inspired UI with live analytics and historical lead tracking.
*   **Professional Architecture:** Reorganized directory structure adhering to industry standards.
*   **Comprehensive Evaluation:** Includes scripts for model performance auditing and out-of-distribution (OOD) testing.

## 📁 Project Structure

```text
LeadSense-AI/
├── api.py                  # FastAPI Backend Entry Point
├── schemas.py              # Pydantic Data Models
├── requirements.txt        # Project Dependencies
├── README.md               # Project Documentation
├── .gitignore              # Git exclusion rules
├── models/                 # Trained model checkpoints
├── static/                 # Frontend assets (HTML, CSS, JS)
├── services/               # Business logic and services
├── inference/              # Lead classification engines
├── data/                   # Datasets and CRM history
├── training/               # Model training scripts
├── evaluation/             # Testing and performance reports
├── preprocessing/          # Data cleaning and splitting
├── docs/                   # Detailed project documentation
└── logs/                   # Execution and audit logs
```

## 🛠 Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/LeadSense-AI.git
    cd LeadSense-AI
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🎮 How to Run

1.  **Start the Backend Server:**
    ```bash
    uvicorn api:app --reload
    ```
2.  **Access the Dashboard:**
    Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.

3.  **Run Evaluation:**
    ```bash
    python evaluation/evaluate.py --mode all
    ```

## 🧪 Testing

Use `test_predict.py` for CLI-based lead classification:
```bash
python test_predict.py
```

## 📄 Documentation

Detailed project notes, technical methodology, and performance metrics can be found in [docs/project_notes.md](docs/project_notes.md).
