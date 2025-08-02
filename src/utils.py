# Example: Load training_config.json
import json
import os

def load_training_config(file_path="training_config.json"):
    config_file = file_path
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


from huggingface_hub import login as hf_login
import os

def login_huggingface(token_env_var="HF_TOKEN"):
    """
    Login to Hugging Face using token from environment variable.
    """
    hf_token = os.getenv(token_env_var)
    if hf_token:
        hf_login(hf_token)
        print("Logged in to Hugging Face Hub.")
    else:
        print(f"Environment variable '{token_env_var}' not found.")
