import os

from src.AI_Banking_Assistant.core.config import Config
from src.AI_Banking_Assistant.mlops import tracking


def test_config_includes_mlop_defaults():
	config = Config()

	assert config.get("model", "name") == "meta-llama/Llama-3.2-3B-Instruct"
	assert config.get("rag", "top_k") == 5
	assert config.get("guardrails", "max_input_length") == 2048
	assert config.get("mlops", "experiment_name") == "AI-Banking-Assistant"


def test_setup_mlflow_uses_tracking_uri_and_dagshub_username(monkeypatch):
	calls = {}
	monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.com/mlflow")
	monkeypatch.setenv("DAGSHUB_USER", "test-user")
	monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)

	monkeypatch.setattr(tracking.mlflow, "set_tracking_uri", lambda uri: calls.setdefault("tracking_uri", uri))
	monkeypatch.setattr(tracking.mlflow, "set_experiment", lambda experiment: calls.setdefault("experiment", experiment))

	tracking.setup_mlflow("Test-Experiment")

	assert calls["tracking_uri"] == "https://example.com/mlflow"
	assert calls["experiment"] == "Test-Experiment"
	assert os.environ["MLFLOW_TRACKING_USERNAME"] == "test-user"
