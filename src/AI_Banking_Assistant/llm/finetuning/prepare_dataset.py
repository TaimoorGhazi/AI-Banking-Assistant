"""
Dataset preparation for fine-tuning.
Converts raw Q&A data into instruction-tuning format for SFTTrainer.
"""

import json
from pathlib import Path
from typing import Dict, List

from src.AI_Banking_Assistant.core.logger import get_logger
from src.AI_Banking_Assistant.core.constants import DATA_FINETUNE_DIR

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


def create_instruction_sample(
    instruction: str,
    response: str,
    context: str = "",
) -> Dict[str, str]:
    """Create a single instruction-tuning sample.
    
    Args:
        instruction: The user question/instruction
        response: The expected model response
        context: Optional context (for RAG training)
        
    Returns:
        Dict with 'instruction', 'context', 'response' keys
    """
    return {
        "instruction": instruction.strip(),
        "context": context.strip(),
        "response": response.strip(),
    }


def format_for_sft(sample: Dict[str, str]) -> str:
    """Format a sample into the SFT training text format.
    
    Args:
        sample: Dict with instruction, context, and response
        
    Returns:
        Formatted training text string
    """
    if sample.get("context"):
        text = (
            f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Context:\n{sample['context']}\n\n"
            f"### Response:\n{sample['response']}"
        )
    else:
        text = (
            f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Response:\n{sample['response']}"
        )
    return text


def load_jsonl(filepath: str) -> List[Dict]:
    """Load samples from a JSONL file."""
    samples = []
    path = Path(filepath)
    if not path.exists():
        logger.warning(f"JSONL file not found: {filepath}")
        return samples

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    logger.info(f"Loaded {len(samples)} samples from {filepath}")
    return samples


def save_jsonl(samples: List[Dict], filepath: str):
    """Save samples to a JSONL file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    logger.info(f"Saved {len(samples)} samples to {filepath}")


def create_banking_dataset() -> List[Dict]:
    """Create a sample banking fine-tuning dataset.
    
    Returns:
        List of instruction-tuning samples
    """
    samples = [
        create_instruction_sample(
            "How do I reset my password?",
            "To reset your password, visit www.securebank.com/reset and enter your account number or registered email. You will receive a verification code via SMS or email. Enter the code, create a new password with at least 8 characters including uppercase, lowercase, number, and special character, then confirm. If you cannot reset online, call 1-800-BANK-HELP or visit your nearest branch with a valid photo ID."
        ),
        create_instruction_sample(
            "What types of accounts do you offer?",
            "SecureBank offers several account types: Basic Checking (no minimum balance, $5/month fee waived with direct deposit), Premium Checking ($1,500 minimum, no fees, 0.05% APY), Savings ($100 minimum, 0.50% APY, 6 withdrawals/month), Money Market ($2,500 minimum, 1.25% APY), and Certificates of Deposit (3 months to 5 years, 2.00% to 4.50% APY)."
        ),
        create_instruction_sample(
            "My card was stolen, what should I do?",
            "If your card is lost or stolen, immediately call our 24/7 hotline at 1-800-CARD-STOP (1-800-227-3786). You can also log into online banking and go to Card Services and block the card. A temporary card can be issued at any branch within 30 minutes, and a permanent replacement will be mailed within 5-7 business days. Please also review your recent transactions for any unauthorized activity."
        ),
        create_instruction_sample(
            "How can I apply for a personal loan?",
            "SecureBank offers personal loans from $1,000 to $50,000 with terms of 12 to 60 months and fixed APR from 5.99% to 17.99% based on creditworthiness. There are no origination fees or prepayment penalties. You will need a valid photo ID, proof of income (last 2 pay stubs or tax returns), proof of address, and your Social Security Number for a credit check."
        ),
        create_instruction_sample(
            "What security measures does SecureBank use?",
            "SecureBank uses multiple layers of security including 256-bit SSL encryption for all online transactions, two-factor authentication via SMS or authenticator app, real-time transaction monitoring and alerts, chip-enabled cards, and biometric login on our mobile app. We never share your information with third parties for marketing, and all data is encrypted at rest and in transit."
        ),
        create_instruction_sample(
            "How do I set up bill pay?",
            "You can set up bill payments through online banking. We support one-time or recurring payments to any company or individual in the US. Electronic payments arrive in 1-2 business days, while check payments take 5-7 business days. You can schedule payments up to 365 days in advance."
        ),
        create_instruction_sample(
            "What are the mobile deposit limits?",
            "With SecureBank Mobile, you can deposit checks using mobile deposit with a daily limit of up to $5,000. Simply open the app, go to deposits, take a photo of the front and back of your check, and submit. The funds are typically available within 1-2 business days."
        ),
        create_instruction_sample(
            "Tell me about your mortgage options.",
            "SecureBank offers fixed-rate mortgages with 15-year and 30-year terms, as well as adjustable-rate mortgages with 5/1 and 7/1 options. The minimum down payment is 3% for first-time homebuyers and 10% standard. Pre-approval takes 1-3 business days, and closing typically takes 30-45 days after approval."
        ),
    ]

    return samples


def prepare_train_val_test(
    samples: List[Dict] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """Split and save dataset into train/val/test JSONL files.
    
    Args:
        samples: List of samples (default: generate banking dataset)
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
    """
    if samples is None:
        samples = create_banking_dataset()

    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    base_dir = PROJECT_ROOT / DATA_FINETUNE_DIR
    save_jsonl(train_samples, str(base_dir / "train.jsonl"))
    save_jsonl(val_samples, str(base_dir / "val.jsonl"))
    save_jsonl(test_samples, str(base_dir / "test.jsonl"))

    logger.info(f"Dataset split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")


if __name__ == "__main__":
    prepare_train_val_test()
    print("Dataset prepared successfully.")
