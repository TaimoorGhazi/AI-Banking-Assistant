# 🏦 SecureBank AI Assistant

> **CS416: Large Language Models · BESE-13 · NUST SEECS**

An LLM-powered customer service chatbot for banking queries, built with a full MLOps pipeline.

---

## 👥 Team Members

| Member | GitHub | Role |
|--------|--------|------|
| Taimoor Ghazi | [@TaimoorGhazi](https://github.com/TaimoorGhazi) | Core, LLM, agents, API, Docker |
| Aldrich Wilder | [@NorthStag](https://github.com/NorthStag) | Preprocessing, guardrails, fine-tuning, Airflow |

---

## 🧠 What This Project Does

An AI banking assistant that:
- Answers customer queries using **RAG** (Retrieval-Augmented Generation)
- Detects and blocks **jailbreak/prompt injection** attempts
- **Anonymizes PII** (names, emails, SSNs, account numbers)
- Supports **real-time document upload** for knowledge base updates
- Tracks experiments with **MLflow** on DagsHub

---

## 🏗️ Architecture

```
[ Streamlit UI ]
      │ HTTP
[ FastAPI Backend ]
      │
[ Guard Rails ]        ← jailbreak detection, PII filter
      │
[ RAG Pipeline ]       ← LangChain + FAISS vector search
      │
[ Llama-3.2-3B ]       ← 4-bit quantized + LoRA fine-tuned
      │
[ Data Layer ]         ← DVC-tracked corpus + FAISS index

[ MLflow → DagsHub ]   ← experiment tracking
[ Airflow ]            ← pipeline orchestration
[ Docker ]             ← containerized deployment
```

---

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| LLM | Llama-3.2-3B-Instruct (4-bit quantized) |
| Fine-Tuning | PEFT (LoRA/QLoRA) + TRL SFTTrainer |
| RAG | LangChain + FAISS |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Agents | LangGraph multi-agent orchestration |
| PII Detection | Microsoft Presidio + spaCy |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| MLOps | DVC + MLflow + DagsHub + Airflow |
| Deployment | Docker + docker-compose |

---

## 📂 Project Structure

```
AI-Banking-Assistant/
├── src/AI_Banking_Assistant/      # Main source code
│   ├── core/                      # Config, logging, constants
│   ├── agents/                    # LangGraph agents
│   ├── llm/                       # Model loading, inference, fine-tuning
│   ├── retrieval/                 # RAG embeddings and FAISS
│   ├── preprocessing/             # Data cleaning, PII anonymization
│   ├── guardrails/                # Safety filters
│   ├── api/                       # FastAPI backend
│   └── ui/                        # Streamlit frontend
├── airflow/dags/                  # Pipeline orchestration
├── data/                          # Data (DVC-tracked)
├── tests/                         # Unit and integration tests
├── config/                        # YAML configurations
├── docker/                        # Dockerfiles
├── docs/                          # Documentation
└── scripts/                       # Utility scripts
```

---

## 🚀 Setup & Running

### Prerequisites
- Python 3.11+
- CUDA GPU (recommended) or CPU
- HuggingFace account with Llama-3.2 access
- DagsHub account

### Quick Start
```bash
git clone https://github.com/TaimoorGhazi/AI-Banking-Assistant.git
cd AI-Banking-Assistant
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env   # Fill in your tokens
```

### Run the App
```bash
# Backend
uvicorn src.AI_Banking_Assistant.api.main:app --reload --port 8000

# Frontend
streamlit run src/AI_Banking_Assistant/ui/app.py
```

---

## 📊 Development Progress

- [x] Phase 1: Project Foundation (structure, configs, core module, DVC, CI/CD)
- [ ] Phase 2: Preprocessing Module (data ingestion, PII, chunking)
- [ ] Phase 3: RAG Retrieval System
- [ ] Phase 4: LLM Module
- [ ] Phase 5: Guardrails
- [ ] Phase 6: LangGraph Agents
- [ ] Phase 7: API & UI
- [ ] Phase 8: MLOps Integration
- [ ] Phase 9: Documentation & Tests

---

## 📬 Contact

Course queries: aabid.msai23seecs@seecs.edu.pk