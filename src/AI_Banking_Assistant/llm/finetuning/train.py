"""
LoRA fine-tuning pipeline with MLflow tracking.
Fine-tunes Llama-3.2-3B using QLoRA and logs to DagsHub/MLflow.
"""

import os
from pathlib import Path
from typing import Optional

from src.AI_Banking_Assistant.core.logger import get_logger
from src.AI_Banking_Assistant.core.config import Config
from src.AI_Banking_Assistant.core.constants import (
    DEFAULT_LORA_RANK,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_BATCH_SIZE,
)

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


def setup_mlflow():
    """Initialize MLflow tracking with DagsHub."""
    config = Config()
    tracking_uri = config.mlflow_tracking_uri
    dagshub_user = config.dagshub_user

    if tracking_uri:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI: {tracking_uri}")

        if dagshub_user:
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
            logger.info(f"DagsHub user: {dagshub_user}")

    return tracking_uri is not None


def train(
    model_name: str = None,
    output_dir: str = None,
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    train_file: str = None,
    val_file: str = None,
):
    """Fine-tune the model using LoRA/QLoRA with SFTTrainer.
    
    Args:
        model_name: Base model to fine-tune
        output_dir: Where to save the LoRA adapter
        lora_rank: LoRA rank (lower = fewer params)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout for LoRA layers
        learning_rate: Training learning rate
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        train_file: Path to training JSONL
        val_file: Path to validation JSONL
    """
    import torch
    from datasets import load_dataset
    from transformers import TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer

    from src.AI_Banking_Assistant.llm.model_loader import load_model
    from src.AI_Banking_Assistant.llm.finetuning.prepare_dataset import format_for_sft

    config = Config()
    model_name = model_name or config.model_name
    output_dir = output_dir or str(PROJECT_ROOT / "src" / "AI_Banking_Assistant" / "llm" / "finetuning" / "output" / "lora_adapter")
    train_file = train_file or str(PROJECT_ROOT / "data" / "finetune" / "train.jsonl")
    val_file = val_file or str(PROJECT_ROOT / "data" / "finetune" / "val.jsonl")

    # Setup MLflow tracking
    use_mlflow = setup_mlflow()

    logger.info(f"Starting LoRA fine-tuning: rank={lora_rank}, lr={learning_rate}, epochs={num_epochs}")

    # Load base model
    model, tokenizer = load_model(model_name, use_quantization=True)

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Load dataset
    data_files = {"train": train_file}
    if Path(val_file).exists():
        data_files["validation"] = val_file

    dataset = load_dataset("json", data_files=data_files)

    def format_sample(example):
        example["text"] = format_for_sft(example)
        return example

    dataset = dataset.map(format_sample)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if "validation" in dataset else "no",
        fp16=torch.cuda.is_available(),
        report_to="mlflow" if use_mlflow else "none",
        optim="paged_adamw_8bit",
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        args=training_args,
        dataset_text_field="text",
        max_seq_length=1024,
    )

    # Log to MLflow
    if use_mlflow:
        import mlflow
        with mlflow.start_run(run_name="lora-finetune"):
            mlflow.log_params({
                "model_name": model_name,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
            })
            trainer.train()
            trainer.save_model(output_dir)
            mlflow.log_artifact(output_dir)
            logger.info(f"Training complete. Model saved to {output_dir}")
    else:
        trainer.train()
        trainer.save_model(output_dir)
        logger.info(f"Training complete. Model saved to {output_dir}")

    return output_dir


if __name__ == "__main__":
    train()
