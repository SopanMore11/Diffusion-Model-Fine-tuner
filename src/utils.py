import json
import os

from huggingface_hub import login as hf_login


def load_training_config(file_path: str = "training_config.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def login_huggingface(token_env_var: str = "HF_TOKEN"):
    """Login to Hugging Face using token from environment variable, if available."""
    hf_token = os.getenv(token_env_var)
    if hf_token:
        hf_login(hf_token)
        print("Logged in to Hugging Face Hub.")
    else:
        print(f"Environment variable '{token_env_var}' not found.")
