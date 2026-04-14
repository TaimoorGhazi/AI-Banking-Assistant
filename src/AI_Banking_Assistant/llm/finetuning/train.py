"""LoRA/QLoRA fine-tuning entrypoint for the SecureBank assistant."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.AI_Banking_Assistant.core.config import Config
from src.AI_Banking_Assistant.core.constants import (  # noqa: E402
    COMPUTE_DTYPE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_RANK,
    DEFAULT_NUM_EPOCHS,
    QUANTIZATION_TYPE,
)
from src.AI_Banking_Assistant.core.logger import get_logger  # noqa: E402
from src.AI_Banking_Assistant.llm.finetuning.prepare_dataset import format_for_sft  # noqa: E402
from src.AI_Banking_Assistant.mlops.tracking import setup_mlflow  # noqa: E402

logger = get_logger(__name__)


def _read_adapter_base_model() -> str | None:
    adapter_config = PROJECT_ROOT / "lora_adapter" / "adapter_config.json"
    if not adapter_config.exists():
        return None
    try:
        with open(adapter_config, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        base = data.get("base_model_name_or_path")
        return str(base).strip() if base else None
    except Exception as exc:
        logger.warning("Could not read adapter_config.json for base model detection: %s", exc)
        return None


def _resolve_model_name(config: Config, explicit_model_name: str | None) -> str:
    """Resolve base model with strong defaults for this repository."""
    if explicit_model_name:
        return explicit_model_name

    env_model = os.getenv("FINETUNE_MODEL_NAME")
    if env_model:
        return env_model.strip()

    cfg_model = config.get("finetuning", "base_model", default=None)
    if cfg_model:
        return str(cfg_model).strip()

    adapter_base = _read_adapter_base_model()
    if adapter_base:
        return adapter_base

    gpu_profile_model = config.get("runtime", "profiles", "gpu_adapter", "model_name", default=None)
    if gpu_profile_model:
        return str(gpu_profile_model).strip()

    return config.model_name


def _normalize_target_modules(value: Iterable[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def train(
    model_name: str = None,
    output_dir: str = None,
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_seq_length: int = 1024,
    gradient_accumulation_steps: int = 4,
    train_file: str = None,
    val_file: str = None,
    target_modules: list[str] | None = None,
    use_4bit: bool = True,
    allow_cpu: bool = False,
) -> str:
    """Fine-tune a causal LLM with LoRA/QLoRA using TRL's SFTTrainer."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from trl import SFTTrainer

    config = Config()
    finetuning_settings = config.get("finetuning", default={}) or {}

    model_name = _resolve_model_name(config, model_name)
    output_dir = output_dir or str(PROJECT_ROOT / "lora_adapter")
    train_file = train_file or str(PROJECT_ROOT / "data" / "finetune" / "train.jsonl")
    val_file = val_file or str(PROJECT_ROOT / "data" / "finetune" / "val.jsonl")

    lora_rank = int(finetuning_settings.get("lora_rank", lora_rank))
    lora_alpha = int(finetuning_settings.get("lora_alpha", lora_alpha))
    lora_dropout = float(finetuning_settings.get("lora_dropout", lora_dropout))
    learning_rate = float(finetuning_settings.get("learning_rate", learning_rate))
    num_epochs = int(finetuning_settings.get("num_epochs", num_epochs))
    batch_size = int(finetuning_settings.get("batch_size", batch_size))
    max_seq_length = int(finetuning_settings.get("max_seq_length", max_seq_length))
    gradient_accumulation_steps = int(finetuning_settings.get("gradient_accumulation_steps", gradient_accumulation_steps))

    if target_modules is None:
        configured_targets = finetuning_settings.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        target_modules = _normalize_target_modules(configured_targets)
    else:
        target_modules = _normalize_target_modules(target_modules)

    if not Path(train_file).exists():
        raise FileNotFoundError(f"Training dataset not found: {train_file}")

    has_cuda = torch.cuda.is_available()
    if not has_cuda and not allow_cpu:
        raise RuntimeError(
            "CUDA is not available. Fine-tuning this model on CPU will be very slow. "
            "Use a GPU machine or pass allow_cpu=True explicitly."
        )

    use_4bit = bool(use_4bit and has_cuda)
    supports_bf16 = bool(has_cuda and torch.cuda.get_device_capability(0)[0] >= 8)
    compute_dtype = getattr(torch, COMPUTE_DTYPE, torch.float16)

    logger.info(
        "Starting fine-tuning with model=%s, 4bit=%s, epochs=%s, batch=%s, grad_accum=%s",
        model_name,
        use_4bit,
        num_epochs,
        batch_size,
        gradient_accumulation_steps,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=config.hf_token,
        trust_remote_code=True,
        cache_dir=config.hf_cache_dir,
        local_files_only=config.hf_local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "token": config.hf_token,
        "trust_remote_code": True,
        "cache_dir": config.hf_cache_dir,
        "local_files_only": config.hf_local_files_only,
    }

    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=QUANTIZATION_TYPE,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    elif has_cuda:
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    logger.info(
        "Trainable parameters: %s / %s (%.2f%%)",
        f"{trainable_params:,}",
        f"{total_params:,}",
        100 * trainable_params / max(total_params, 1),
    )

    data_files = {"train": train_file}
    if Path(val_file).exists():
        data_files["validation"] = val_file

    dataset = load_dataset("json", data_files=data_files)
    dataset = dataset.map(lambda example: {"text": format_for_sft(example)})

    use_mlflow = setup_mlflow()

    training_args_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "gradient_checkpointing": True,
        "fp16": has_cuda and not supports_bf16,
        "bf16": supports_bf16,
        "optim": "paged_adamw_8bit" if use_4bit else "adamw_torch",
        "report_to": "mlflow" if use_mlflow else "none",
    }

    evaluation_mode = "epoch" if "validation" in dataset else "no"
    training_args_signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in training_args_signature.parameters:
        training_args_kwargs["evaluation_strategy"] = evaluation_mode
    elif "eval_strategy" in training_args_signature.parameters:
        training_args_kwargs["eval_strategy"] = evaluation_mode

    training_args = TrainingArguments(**training_args_kwargs)

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset.get("validation"),
        "args": training_args,
    }

    sft_signature = inspect.signature(SFTTrainer.__init__)
    if "tokenizer" in sft_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in sft_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    if "dataset_text_field" in sft_signature.parameters:
        trainer_kwargs["dataset_text_field"] = "text"

    if "max_seq_length" in sft_signature.parameters:
        trainer_kwargs["max_seq_length"] = max_seq_length

    if "packing" in sft_signature.parameters:
        trainer_kwargs["packing"] = False

    trainer = SFTTrainer(**trainer_kwargs)

    if use_mlflow:
        import mlflow

        with mlflow.start_run(run_name="lora-finetune"):
            mlflow.log_params(
                {
                    "model_name": model_name,
                    "lora_rank": lora_rank,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "max_seq_length": max_seq_length,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "use_4bit": use_4bit,
                    "target_modules": ",".join(target_modules),
                }
            )
            trainer.train()
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            mlflow.log_artifact(output_dir)
    else:
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    meta = {
        "base_model": model_name,
        "target_modules": target_modules,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "max_seq_length": max_seq_length,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "use_4bit": use_4bit,
        "train_file": train_file,
        "val_file": val_file if Path(val_file).exists() else None,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "training_config.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    logger.info("Training complete. Adapter saved to %s", output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SecureBank LLM with LoRA/QLoRA")
    parser.add_argument("--model-name", default=None, help="Base model to fine-tune")
    parser.add_argument("--output-dir", default=None, help="Where to save the LoRA adapter")
    parser.add_argument("--train-file", default=None, help="Path to training JSONL")
    parser.add_argument("--val-file", default=None, help="Path to validation JSONL")
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument(
        "--target-modules",
        default=None,
        help="Comma-separated target modules, e.g. q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit QLoRA and use full precision LoRA")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU fine-tuning (very slow)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = train(
        model_name=args.model_name,
        output_dir=args.output_dir,
        lora_rank=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        gradient_accumulation_steps=args.grad_accum,
        train_file=args.train_file,
        val_file=args.val_file,
        target_modules=_normalize_target_modules(args.target_modules) if args.target_modules else None,
        use_4bit=not args.no_4bit,
        allow_cpu=args.allow_cpu,
    )
    print(f"Adapter saved to: {output}")
