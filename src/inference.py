import torch
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

def create_inference_pipeline(model_name: str, lora_path: str, lora_rank: int = 16, lora_alpha: int = 16, dtype=torch.float16, device: str = "cuda"):
    """
    Loads models and applies LoRA weights to create an inference pipeline.

    Args:
        model_name (str): Name of the base Stable Diffusion model.
        lora_path (str): Path to the saved LoRA weights.
        lora_rank (int): LoRA rank used during training.
        lora_alpha (int): LoRA alpha used during training.
        dtype (torch.dtype): Data type for the models.
        device (str): Device to run inference on.

    Returns:
        StableDiffusionPipeline: The configured inference pipeline.
    """
    device = torch.device(device)

    # Load base UNet and apply LoRA configuration
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", torch_dtype=dtype)
    target_modules = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2", "proj_out"]
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none"
    )
    unet = get_peft_model(unet, lora_config)

    # Load saved LoRA weights into the model
    lora_state_dict = load_file(lora_path)
    set_peft_model_state_dict(unet, lora_state_dict)

    # Load other components
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=dtype)
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler", torch_dtype=dtype)

    # Create and return the pipeline
    inference_pipeline = StableDiffusionPipeline(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    return inference_pipeline.to(device=device, dtype=dtype)