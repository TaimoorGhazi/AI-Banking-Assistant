import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "AI_Banking_Assistant"

list_of_files = [
    # ===== Core Module =====
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/core/__init__.py",
    f"src/{project_name}/core/config.py",
    f"src/{project_name}/core/logger.py",
    f"src/{project_name}/core/exceptions.py",
    f"src/{project_name}/core/constants.py",

    # ===== Agents Module (LangGraph) =====
    f"src/{project_name}/agents/__init__.py",
    f"src/{project_name}/agents/base_agent.py",
    f"src/{project_name}/agents/rag_agent.py",
    f"src/{project_name}/agents/response_agent.py",
    f"src/{project_name}/agents/guardrail_agent.py",
    f"src/{project_name}/agents/orchestrator.py",

    # ===== LLM Module =====
    f"src/{project_name}/llm/__init__.py",
    f"src/{project_name}/llm/model_loader.py",
    f"src/{project_name}/llm/inference.py",
    f"src/{project_name}/llm/prompt_templates.py",
    f"src/{project_name}/llm/finetuning/__init__.py",
    f"src/{project_name}/llm/finetuning/prepare_dataset.py",
    f"src/{project_name}/llm/finetuning/train.py",
    f"src/{project_name}/llm/finetuning/evaluate.py",
    f"src/{project_name}/llm/finetuning/config.yaml",

    # ===== Retrieval Module (RAG) =====
    f"src/{project_name}/retrieval/__init__.py",
    f"src/{project_name}/retrieval/embeddings.py",
    f"src/{project_name}/retrieval/vector_store.py",
    f"src/{project_name}/retrieval/retriever.py",
    f"src/{project_name}/retrieval/indexer.py",

    # ===== Preprocessing Module =====
    f"src/{project_name}/preprocessing/__init__.py",
    f"src/{project_name}/preprocessing/data_ingestion.py",
    f"src/{project_name}/preprocessing/text_cleaner.py",
    f"src/{project_name}/preprocessing/pii_anonymizer.py",
    f"src/{project_name}/preprocessing/chunking.py",
    f"src/{project_name}/preprocessing/validators.py",

    # ===== Guardrails Module =====
    f"src/{project_name}/guardrails/__init__.py",
    f"src/{project_name}/guardrails/input_filter.py",
    f"src/{project_name}/guardrails/output_filter.py",
    f"src/{project_name}/guardrails/pii_detector.py",
    f"src/{project_name}/guardrails/prompt_injection_detector.py",
    f"src/{project_name}/guardrails/audit_logger.py",
    f"src/{project_name}/guardrails/rules.yaml",

    # ===== API Module (FastAPI) =====
    f"src/{project_name}/api/__init__.py",
    f"src/{project_name}/api/main.py",
    f"src/{project_name}/api/routes/__init__.py",
    f"src/{project_name}/api/routes/chat.py",
    f"src/{project_name}/api/routes/upload.py",
    f"src/{project_name}/api/routes/health.py",
    f"src/{project_name}/api/routes/admin.py",
    f"src/{project_name}/api/schemas/__init__.py",
    f"src/{project_name}/api/schemas/request.py",
    f"src/{project_name}/api/schemas/response.py",
    f"src/{project_name}/api/schemas/chat.py",
    f"src/{project_name}/api/middleware/__init__.py",
    f"src/{project_name}/api/middleware/rate_limiter.py",
    f"src/{project_name}/api/middleware/auth.py",
    f"src/{project_name}/api/middleware/cors.py",

    # ===== UI Module (Streamlit) =====
    f"src/{project_name}/ui/__init__.py",
    f"src/{project_name}/ui/app.py",
    f"src/{project_name}/ui/components/__init__.py",
    f"src/{project_name}/ui/components/chat_interface.py",
    f"src/{project_name}/ui/components/sidebar.py",
    f"src/{project_name}/ui/components/upload_widget.py",
    f"src/{project_name}/ui/utils/__init__.py",
    f"src/{project_name}/ui/utils/session_state.py",
    f"src/{project_name}/ui/utils/api_client.py",

    # ===== Utils Module =====
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/file_handler.py",
    f"src/{project_name}/utils/helpers.py",
    f"src/{project_name}/utils/metrics.py",
    f"src/{project_name}/utils/text_utils.py",

    # ===== Airflow =====
    "airflow/dags/__init__.py",
    "airflow/dags/preprocessing_dag.py",
    "airflow/dags/embedding_dag.py",
    "airflow/dags/training_dag.py",
    "airflow/dags/ingestion_dag.py",
    "airflow/plugins/__init__.py",
    "airflow/configs/airflow.cfg",

    # ===== GitHub Actions =====
    ".github/workflows/ci.yml",
    ".github/workflows/docker-publish.yml",
    ".github/workflows/dvc-sync.yml",

    # ===== Data Directories =====
    "data/raw/.gitkeep",
    "data/raw/sample_data.txt",
    "data/processed/.gitkeep",
    "data/finetune/train.jsonl",
    "data/finetune/val.jsonl",
    "data/finetune/test.jsonl",
    "data/faiss_index/.gitkeep",

    # ===== Tests =====
    "tests/__init__.py",
    "tests/conftest.py",
    "tests/unit/__init__.py",
    "tests/unit/test_preprocessing.py",
    "tests/unit/test_embeddings.py",
    "tests/unit/test_retrieval.py",
    "tests/unit/test_guardrails.py",
    "tests/unit/test_llm.py",
    "tests/unit/test_agents.py",
    "tests/integration/__init__.py",
    "tests/integration/test_rag_pipeline.py",
    "tests/integration/test_api.py",
    "tests/integration/test_end_to_end.py",
    "tests/fixtures/__init__.py",
    "tests/fixtures/sample_documents.py",
    "tests/fixtures/mock_responses.py",

    # ===== Documentation =====
    "docs/architecture/.gitkeep",
    "docs/guides/setup_guide.md",
    "docs/guides/deployment_guide.md",
    "docs/guides/fine_tuning_guide.md",
    "docs/guides/api_documentation.md",
    "docs/development/contributing.md",
    "docs/development/coding_standards.md",
    "docs/development/testing_guide.md",

    # ===== Config =====
    "config/settings.yaml",
    "config/model_config.yaml",
    "config/rag_config.yaml",
    "config/guardrails_config.yaml",

    # ===== Docker =====
    "docker/Dockerfile.api",
    "docker/Dockerfile.ui",
    "docker/Dockerfile.airflow",
    "docker/docker-compose.yml",

    # ===== Notebooks =====
    "notebooks/01_data_exploration.ipynb",
    "notebooks/02_embedding_experiments.ipynb",
    "notebooks/03_rag_testing.ipynb",
    "notebooks/04_fine_tuning_analysis.ipynb",

    # ===== Scripts =====
    "scripts/setup_env.sh",
    "scripts/download_models.py",
    "scripts/init_dvc.sh",
    "scripts/build_index.py",
    "scripts/run_tests.sh",

    # ===== MLflow =====
    "mlflow/mlruns/.gitkeep",
    "mlflow/artifacts/.gitkeep",

    # ===== Logs =====
    "logs/.gitkeep",

    # ===== Root Files =====
    "setup.py",
    "CHANGELOG.md",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")