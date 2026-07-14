# CognixAI — Healthcare Explainable AI

CognixAI is an explainable-AI dashboard for chronic disease progression. It combines an
XGBoost prediction model, SHAP-based explainability, automatic patient phenotyping, and a
local LLM chatbot to help clinicians understand *why* a patient's cognitive score is predicted
to change — and *what could change it*.

The system pairs a Flask REST/SSE API (`api.py`) with a React + TypeScript dashboard
(`frontend/`), and uses [Ollama](https://ollama.com) to run the chat/reasoning LLM entirely
locally — no external API calls, no patient data leaving the machine.

## Features

- **XGBoost Prediction** — regression model forecasts each patient's future cognitive score
  and trajectory from longitudinal visit data.
- **SHAP Explainability** — per-patient and cohort-level SHAP values explain which features
  drive each prediction, with beeswarm/bar visualizations.
- **Patient Phenotyping** — patients are automatically clustered into clinically meaningful
  phenotype groups based on their dominant SHAP factors.
- **LLM Chatbot (Ollama)** — a locally-hosted LLM answers natural-language questions about a
  patient, grounded in their SHAP factors, phenotype, and peer comparisons.
- **What-If Simulation Engine** — explore hypothetical feature changes (e.g. "what if BMI
  dropped by 5 points?") and see the model's predicted impact in real time.
- **React Dashboard** — a TypeScript + Vite single-page app for browsing patients,
  explanations, phenotypes, and the chat interface.

## Demo Videos

### 🎥 Overview
A walkthrough of the dashboard, peer comparisons, and what-if simulation.

[▶ Watch Demo](Demo/Dashboard.webm)

### 🎥 Chatbot
A walkthrough of the AI chatbot assistant.

[▶ Watch Demo](Demo/Chatbot.webm)

## Architecture

The pipeline runs once (or whenever the source data or model logic changes) to produce the
artifacts served by the application at runtime:

```
chronic_disease_progression.csv
        │
        ▼
pipeline.py                    cleaning, feature engineering, train/test split
        │
        ▼
xgboost_and_shap.py             trains the model, computes SHAP values
        │                       writes predictions_*.csv, shap_values.pkl,
        │                       feature_importance.csv, model_report.txt
        ▼
generate_shap_phenotype.py      clusters patients into phenotypes
        │                       builds the RAG knowledge base for the chatbot
        ▼
outputs/                        generated CSVs, plots, and RAG JSON
        │
        ▼
api.py  ── Flask ──►  frontend/dist            (built React app)
        │
        ▼
Ollama (local LLM) ── chat replies / simulation explanations
```

**Supporting modules:**

| Module | Responsibility |
|---|---|
| `prompt_builder.py` | Assembles per-patient context and the system prompt for the LLM |
| `simulation_engine.py` | Runs "what if" feature-change simulations through the trained model |
| `reasoning.py` | Clinical reasoning layer — actionable factors, cautions, peer comparisons |
| `llm_backend.py` | Thin client for the Ollama API |
| `llm_chatbot.py` | Standalone CLI chatbot (same backend, no web UI) — useful for quick testing |

## Repository Structure

```
Healthcare/
├── api.py                          Flask REST + SSE API, serves frontend/dist
├── pipeline.py                     Data cleaning / feature engineering
├── xgboost_and_shap.py             Model training + SHAP computation
├── generate_shap_phenotype.py      Phenotype clustering + RAG knowledge base
├── prompt_builder.py               Per-patient LLM context / system prompt
├── reasoning.py                    Clinical reasoning (actionable factors, cautions)
├── simulation_engine.py            "What if" simulation engine
├── llm_backend.py                  Ollama client
├── llm_chatbot.py                  CLI chatbot (no web UI)
├── start.sh                        Start script (see Running, below)
├── requirements.txt                Python dependencies
├── chronic_disease_progression.csv Raw source data
├── outputs/                        Generated model artifacts (git-ignored)
└── frontend/                       React + TypeScript + Vite dashboard
    ├── src/                        Application source
    ├── public/                     Static assets
    └── dist/                       Production build output (generated)
```

## Prerequisites

- Python 3.10
- Node.js with `npm` (for the frontend build)
- [Ollama](https://ollama.com), running locally with a pulled model (default `llama3.1:8b`)

## Installation

```bash
# 1. Clone the repository
git clone https://git.invlab.live/internship/intnership_sredha-gopan_311133.git
cd intnership_sredha-gopan_311133

# 2. Python dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Ollama (in a separate terminal)
ollama serve
ollama pull llama3.1:8b

# 4. Frontend dependencies
cd frontend
npm install
cd ..
```

### Generate model outputs

Run once on first setup, and again any time the source CSV or model/pipeline logic changes.
`outputs/` is git-ignored — everything in it is reproducible from the CSV.

```bash
python3 pipeline.py                  # optional: sanity-check the preprocessing step
python3 xgboost_and_shap.py          # trains model, writes predictions_*.csv, shap_values.pkl, etc.
python3 generate_shap_phenotype.py   # builds phenotypes + RAG knowledge base
```

## Running

```bash
./start.sh            # builds the frontend if needed, then serves everything from :5000
./start.sh --dev       # Flask API only — run the frontend separately for hot reload:
                       #   cd frontend && npm run dev   (proxies /api to :5000, served on :5173)
```

Once running:

| Endpoint | URL |
|---|---|
| Dashboard | http://localhost:5000 |
| API | http://localhost:5000/api |
| Health check | http://localhost:5000/api/health |

## Deployment

There is no Dockerfile or CI pipeline yet — deployment is manual. The application is a single
Flask process serving both the API and the pre-built static frontend, with a dependency on a
reachable Ollama instance.

### Prerequisites

- A host with Python 3.10, Node.js, and Ollama installed (or Ollama reachable over the network).

### Installation

```bash
git clone https://git.invlab.live/internship/intnership_sredha-gopan_311133.git
cd intnership_sredha-gopan_311133
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### Generate model outputs

Only needed once, or whenever the CSV or model/phenotype logic changes — these artifacts are
git-ignored, not committed.

```bash
python3 xgboost_and_shap.py
python3 generate_shap_phenotype.py
```

### Build frontend

Compiles the React app into static assets that Flask will serve.

```bash
cd frontend && npm run build && cd ..
```

`start.sh` runs this automatically if `frontend/dist/index.html` is missing.

### Run application

```bash
./start.sh
```

This runs `python3 api.py`, which binds to `0.0.0.0:5000` and serves both the SPA and the API
from a single process.

> **Note:** `app.run(...)` in `api.py` is Flask's development server, not a production-grade
> WSGI server. For production deployments, run the app behind **Gunicorn** (or Waitress) with
> **Nginx** as a reverse proxy for TLS termination, static-file caching, and a stable public
> port — and manage the process with a supervisor such as systemd, supervisord, or pm2.

### Verify deployment

```bash
curl http://<host>:5000/api/health
```

A healthy deployment reports `"status": "ok"` and a non-zero `patients_loaded` count.

### Redeployment

- **Frontend-only change:** `cd frontend && npm run build && cd ..`, then restart `api.py`.
- **Backend or model logic change:** re-run the *Generate model outputs* and *Build frontend*
  steps above, then restart.
- Restarting the process simply re-reads `outputs/` at startup — regeneration is only required
  when the CSV or the model/phenotype scripts themselves changed.

## Environment Variables

All variables are optional; sensible defaults are defined in `llm_backend.py`.

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `llama3.1:8b` | Model used for chat and intro generation |
| `OLLAMA_TIMEOUT_S` | `120` | Per-request timeout, in seconds |
| `OLLAMA_MAX_RETRIES` | `2` | Retries on timeout or 5xx server errors |

## Future Improvements

- Integrate real, IRB-approved clinical datasets in place of the synthetic sample data.
- Upgrade to a stronger LLM (larger local model or hosted frontier model) for richer reasoning.
- Ground chatbot responses in external medical knowledge sources (e.g. clinical guidelines,
  literature retrieval) rather than SHAP context alone.
- Extend prediction to multiple outcomes beyond cognitive score (e.g. comorbidity risk,
  hospitalization likelihood).
- Add Docker support for reproducible, one-command deployment.
- Add authentication and user management for multi-clinician / multi-tenant use.

## Disclaimer

This project is a research prototype intended for educational purposes. Predictions,
explanations, and simulations are generated by machine learning models and should not be
interpreted as medical advice or used for clinical decision-making without professional
validation.
