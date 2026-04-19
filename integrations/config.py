"""
Central config for training, logging, and deployment.
Override with environment variables or a local .env (not committed).

Before final submission, set CODEDRIFT_HF_* and WANDB_* so logs and metadata
do not show placeholder text.
"""

import os

# --- Models & training ---------------------------------------------------------
DEFAULT_MODEL_ID = os.environ.get(
    "CODEDRIFT_MODEL_ID",
    "unsloth/Qwen2.5-1.5B-Instruct",
)
OUTPUT_DIR = os.environ.get("CODEDRIFT_OUTPUT_DIR", "./codedrift_output")

# --- Hugging Face (empty until you set env vars) -------------------------------
HF_DATASET_REPO = os.environ.get("CODEDRIFT_HF_DATASET_REPO", "")
HF_MODEL_REPO = os.environ.get("CODEDRIFT_HF_MODEL_REPO", "")
HF_SPACE_URL = os.environ.get("CODEDRIFT_HF_SPACE_URL", "")

# --- Weights & Biases / experiment tracking ---------------------------------
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")

# --- OpenEnv server -----------------------------------------------------------
OPENENV_HOST = os.environ.get("OPENENV_HOST", "0.0.0.0")
OPENENV_PORT = int(os.environ.get("OPENENV_PORT", "8000"))
