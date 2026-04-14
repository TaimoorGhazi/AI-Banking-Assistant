# SecureBank AI - NUST Bank Customer Service Agent

Production-grade LLM-powered banking assistant developed for the NUST SEECS 8th Semester LLM course.

This project combines Retrieval-Augmented Generation (RAG), guardrails, agent orchestration, and a full API+UI workflow to answer banking queries with grounded context.

## Team

| Member Name | GitHub | Project Identity in Repo | Primary Contributions |
|---|---|---|---|
| Ahsan Taimoor Ghazi | [@TaimoorGhazi](https://github.com/TaimoorGhazi) | Taimoor Ghazi | Core architecture, RAG pipeline, API/UI integration, repo cleanup, index/data workflows |
| Ahmad Shehroz | [@NorthStag](https://github.com/NorthStag) | NorthStag | Guardrails, agent orchestration, fine-tuning pipeline work, module-level implementation |

Contribution mapping is based on commit history on `feature/guardrails`.

## What This Project Does

- Answers customer banking questions using RAG-grounded responses.
- Applies input and output guardrails for safer interactions.
- Detects and sanitizes sensitive content (PII).
- Supports document upload and index rebuild for iterative knowledge updates.
- Exposes a FastAPI backend and a Streamlit chat interface.

## System Architecture

### High-Level Flow

```text
User -> Streamlit UI -> FastAPI /chat -> LangGraph Orchestrator
                                      -> Input Guardrail Check
                                      -> FAISS Retrieval (top-k)
                                      -> LLM Response Generation
                                      -> Output Sanitization
                                      -> Final Response to UI
```

### Agent Orchestration

The orchestrator in `src/AI_Banking_Assistant/agents/orchestrator.py` executes:

1. Guardrail inspection
2. Retrieval step
3. Response generation
4. Output guard/sanitization

If input is blocked, a safe fallback response is returned.

## Tech Stack (Implemented)

- LLM Runtime: Hugging Face Transformers + PEFT
- Base Models in Config:
  - CPU profile: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
  - GPU profile: `meta-llama/Llama-3.2-3B-Instruct`
- Fine-tuning: LoRA/QLoRA training pipeline in `src/AI_Banking_Assistant/llm/finetuning/train.py`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- Vector Store: FAISS (`IndexFlatL2`) with persistent index on disk
- Retrieval: Dense retrieval from FAISS, configurable `top_k`
- Orchestration: LangGraph state graph
- Guardrails: Input filtering, prompt injection checks, PII redaction, audit logging
- Backend: FastAPI + Uvicorn
- Frontend: Streamlit
- MLOps Utilities: DVC + MLflow + DagsHub integration hooks

## Repository Structure

```text
AI-Banking-Assistant/
|- src/AI_Banking_Assistant/
|  |- agents/
|  |- api/
|  |- core/
|  |- guardrails/
|  |- llm/
|  |- preprocessing/
|  |- retrieval/
|  |- ui/
|  +- utils/
|- config/
|- data/
|  |- raw/
|  |- processed/
|  |- finetune/
|  +- faiss_index/
|- scripts/
|- tests/
|- docs/
+- README.md
```

## Data Sources and Indexing

### Current Raw Sources

From `data/raw/`:

- `NUST Bank-Product-Knowledge (1).xlsx`
- `nust_mobile_banking_faqs.json`
- `sample_data.txt`
- `realtime_rag_5_questions.txt`
- `realtime_update_demo.txt`

### Ingestion Support

Supported file types are defined in `src/AI_Banking_Assistant/core/constants.py`:

- `.txt`, `.pdf`, `.docx`, `.json`, `.jsonl`, `.xlsx`

### RAG Configuration

From `config/rag_config.yaml`:

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Chunk size: `512`
- Chunk overlap: `50`
- Retrieval top-k: `5`
- Index path: `data/faiss_index`

### Current Index Snapshot

Latest local build loaded:

- FAISS vectors: `385`

(Loaded via `src/AI_Banking_Assistant/retrieval/vector_store.py`)

## Guardrails and Safety

From `config/guardrails_config.yaml`:

- Max input length: `2048`
- Max output length: `4096`
- PII redaction enabled: `true`
- Prompt injection blocking enabled: `true`
- Audit log path: `logs/security_audit.log`

Blocked pattern examples include:

- `ignore previous instructions`
- `system prompt`
- `developer message`

## Fine-Tuning (Implemented)

Fine-tuning pipeline exists and is usable from:

- `src/AI_Banking_Assistant/llm/finetuning/train.py`

Key characteristics:

- LoRA/QLoRA support
- 4-bit mode when CUDA is available
- Configurable LoRA rank/alpha/dropout
- MLflow logging integration when configured
- Outputs adapter and training metadata (`training_config.json`)

Dataset files expected under:

- `data/finetune/train.jsonl`
- `data/finetune/val.jsonl`
- `data/finetune/test.jsonl`

## API Endpoints

Implemented routes:

- `GET /` : service status
- `GET /health` : service + model/index health
- `POST /chat` : chat inference endpoint
- `POST /upload_doc` : upload document and optional index rebuild
- `POST /admin/rebuild-index` : rebuild FAISS index from raw data

## Setup and Run

### Prerequisites

- Python 3.11+
- Windows/Linux/macOS (tested in local virtual environment workflow)
- Optional CUDA-capable GPU for faster model loading/fine-tuning

### Installation

```bash
# clone
git clone https://github.com/TaimoorGhazi/AI-Banking-Assistant.git
cd AI-Banking-Assistant

# create venv (Windows)
python -m venv .venv
.venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# optional but recommended for Presidio pipeline quality
python -m spacy download en_core_web_sm
```

### Environment Configuration

Copy and edit environment template:

```bash
copy .env.example .env
```

(Use `cp` instead of `copy` on Linux/macOS.)

### Build / Rebuild Index

```bash
python scripts/build_index.py
```

### Run Backend

```bash
python -m uvicorn src.AI_Banking_Assistant.api.main:app --host 127.0.0.1 --port 8000
```

### Run Frontend

```bash
python -m streamlit run src/AI_Banking_Assistant/ui/app.py --server.address 127.0.0.1 --server.port 8501
```

## Testing

Test framework is configured in `pyproject.toml` and dependencies include pytest tooling.

Run all tests:

```bash
pytest tests/ -v
```

Or via helper script:

```bash
bash scripts/run_tests.sh
```

## Evaluation Status

- Practical QA benchmark file generated: `evaluation_100_questions_answers.md`
- Retrieval/index validation and health checks have been run during development.
- Advanced benchmark section (for example Ragas-style full metric table) can be added once final evidence artifacts are finalized.

## License

Academic project for NUST University LLM course.
