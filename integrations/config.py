"""
Central placeholders for training, logging, and deployment.
Override with environment variables or a local .env (not committed).
"""

import os

# --- Models & training ---------------------------------------------------------
DEFAULT_MODEL_ID = os.environ.get(
    "CODEDRIFT_MODEL_ID",
    "unsloth/Qwen2.5-1.5B-Instruct",
)
OUTPUT_DIR = os.environ.get("CODEDRIFT_OUTPUT_DIR", "./codedrift_output")

# --- Hugging Face -------------------------------------------------------------
HF_DATASET_REPO = os.environ.get(
    "CODEDRIFT_HF_DATASET_REPO",
    "PLACEHOLDER_ORG/codedrift-arena-dataset",
)
HF_MODEL_REPO = os.environ.get(
    "CODEDRIFT_HF_MODEL_REPO",
    "PLACEHOLDER_ORG/codedrift-reviewer-lora",
)
HF_SPACE_URL = os.environ.get(
    "CODEDRIFT_HF_SPACE_URL",
    "https://huggingface.co/spaces/PLACEHOLDER_ORG/codedrift-arena",
)

# --- Weights & Biases / experiment tracking ---------------------------------
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "codedrift-arena-PLACEHOLDER")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "PLACEHOLDER_TEAM")

# --- OpenEnv server -----------------------------------------------------------
OPENENV_HOST = os.environ.get("OPENENV_HOST", "0.0.0.0")
OPENENV_PORT = int(os.environ.get("OPENENV_PORT", "8000"))
