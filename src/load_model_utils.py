import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
import gc

def load_models(model_name: str, lora_rank: int = 16, lora_alpha: int = 16, dtype=torch.float16, device: str = "cuda"):
    """
    Load all required models for Stable Diffusion training and apply LoRA.
    
    Args:
        model_name (str): The name of the pre-trained model to use.
        lora_rank (int): The rank of the LoRA matrices.
        lora_alpha (int): The alpha parameter for LoRA.
        dtype (torch.dtype): The data type for the models.
        device (str): The device to run the models on.

    Returns:
        tuple: A tuple containing the configured unet, vae, text_encoder, and tokenizer.
    """
    device = torch.device(device)

    print("Loading UNet2DConditionModel")
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", torch_dtype=dtype)
    unet.enable_gradient_checkpointing()

    print("Applying LoRA to UNet2DConditionModel")
    target_modules = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2", "proj_out"]
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none"
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    print("Loading AutoencoderKL")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=dtype)
    vae.eval()

    print("Loading CLIPTextModel")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder.eval()

    print("Loading CLIPTokenizer")
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")

    print("Moving models to GPU")
    unet = unet.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    return unet, vae, text_encoder, tokenizer

def flush_memory():
    """Flushes GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_optimizer(model, learning_rate: float = 1e-4):
    """
    Initializes the AdamW 8-bit optimizer.

    Args:
        model (torch.nn.Module): The model for which to create the optimizer.
        learning_rate (float): The learning rate.

    Returns:
        bnb.optim.AdamW8bit: The AdamW 8-bit optimizer.
    """
    return bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)

def get_scheduler(model_name: str, dtype=torch.float16):
    """
    Loads a DDPMScheduler from a pre-trained model.

    Args:
        model_name (str): The name of the pre-trained model.
        dtype (torch.dtype): The data type for the scheduler.

    Returns:
        DDPMScheduler: The noise scheduler.
    """
    print("Loading Noise Scheduler")
    return DDPMScheduler.from_pretrained(model_name, subfolder="scheduler", torch_dtype=dtype)