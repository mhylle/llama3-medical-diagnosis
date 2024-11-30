"""
Configuration settings for the medical diagnosis model fine-tuning
"""

import os
from typing import Dict, Any

# Model Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8b"
OUTPUT_DIR = "medical_diagnosis_model_llama3"
MAX_SEQ_LENGTH = 4096

# Training Configuration
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 6,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "weight_decay": 0.001,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
}

# LoRA Configuration
LORA_CONFIG = {
    "r": 64,
    "lora_alpha": 16,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "w1",
        "w2",
        "w3",
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# Dataset Configuration
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "medical_qa": {
        "path": os.path.join("data", "medical_qa_dataset.json"),
        "preprocessing_fn": "preprocess_medical_qa"
    },
    "mimic": {
        "path": os.path.join("data", "mimic_processed.json"),
        "preprocessing_fn": "preprocess_mimic"
    },
    "healthcaremagic": {
        "path": os.path.join("data", "healthcaremagic_100k.json"),
        "preprocessing_fn": "preprocess_healthcaremagic"
    }
}

# Validation Configuration
VALIDATION_CONFIG = {
    "validation_split": 0.1,
    "seed": 42,
    "metrics": ["accuracy", "f1", "precision", "recall"]
}

# Logging Configuration
LOGGING_CONFIG = {
    "log_level": "INFO",
    "save_strategy": "epoch",
    "save_total_limit": 3,
    "logging_steps": 10,
    "report_to": ["tensorboard", "wandb"]
}