"""
Model evaluation for fine-tuned models.
Compares base vs fine-tuned performance using standard metrics.
"""

from typing import Dict, List

from src.AI_Banking_Assistant.core.logger import get_logger

logger = get_logger(__name__)


def compute_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores for model outputs.
    
    Args:
        predictions: Model-generated responses
        references: Ground truth responses
        
    Returns:
        Dict with rouge1, rouge2, rougeL scores
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        for pred, ref in zip(predictions, references):
            result = scorer.score(ref, pred)
            for key in scores:
                scores[key] += result[key].fmeasure

        n = len(predictions)
        return {k: v / n for k, v in scores.items()}

    except ImportError:
        logger.warning("rouge-score not installed")
        return {}


def compute_bleu_score(predictions: List[str], references: List[str]) -> float:
    """Compute BLEU score for model outputs."""
    try:
        from sacrebleu import corpus_bleu
        result = corpus_bleu(predictions, [references])
        return result.score
    except ImportError:
        logger.warning("sacrebleu not installed")
        return 0.0


def evaluate_model(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Run full evaluation suite.
    
    Returns:
        Dict with all computed metrics
    """
    metrics = {}
    metrics.update(compute_rouge_scores(predictions, references))
    metrics["bleu"] = compute_bleu_score(predictions, references)
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics
