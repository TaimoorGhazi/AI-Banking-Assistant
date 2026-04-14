"""MLflow helper utilities used by training and pipeline scripts."""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

import mlflow

from src.AI_Banking_Assistant.core.config import Config
from src.AI_Banking_Assistant.core.logger import get_logger

logger = get_logger(__name__)


def _get_tracking_uri(config: Config) -> Optional[str]:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config.get("mlops", "tracking_uri")
    return tracking_uri or None


def setup_mlflow(experiment_name: Optional[str] = None) -> None:
    """Configure MLflow to use the repository defaults and active environment."""
    config = Config()
    tracking_uri = _get_tracking_uri(config)
    dagshub_user = os.getenv("DAGSHUB_USER") or config.dagshub_user

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI set to %s", tracking_uri)

    if dagshub_user:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
        logger.info("MLflow tracking username set for DagsHub user %s", dagshub_user)

    experiment = experiment_name or config.mlflow_experiment_name
    mlflow.set_experiment(experiment)
    logger.info("MLflow experiment set to %s", experiment)


def log_training_metrics(metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
    """Log a mapping of metrics to MLflow if a run is active."""
    if not mlflow.active_run():
        logger.debug("Skipping MLflow metric logging because no run is active.")
        return

    safe_metrics = {key: value for key, value in metrics.items() if value is not None}
    if not safe_metrics:
        return

    mlflow.log_metrics(safe_metrics, step=step)


def log_pipeline_metadata(metadata: Mapping[str, Any]) -> None:
    """Log pipeline metadata as MLflow parameters."""
    if not mlflow.active_run():
        logger.debug("Skipping MLflow parameter logging because no run is active.")
        return

    safe_metadata = {key: value for key, value in metadata.items() if value is not None}
    if not safe_metadata:
        return

    mlflow.log_params(safe_metadata)
