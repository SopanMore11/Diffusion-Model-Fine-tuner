from train_flux import train_flux_model
from train_sd import train_sd_model
from utils import login_huggingface

login_huggingface()


def train_model(model_name: str, images_directory: str, hf_api_key: str, model_type: str = "sd"):
    """Dispatch to the correct training function based on model_type."""
    if model_type == "flux":
        train_flux_model(model_name, images_directory, hf_api_key)
    else:
        train_sd_model(model_name, images_directory, hf_api_key)
