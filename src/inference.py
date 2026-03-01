import torch
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer


def create_inference_pipeline(
    model_name: str,
    lora_path: str,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    dtype=torch.float16,
    device: str = "cuda",
):
    """Load models and apply LoRA weights to create an inference pipeline."""
    device = torch.device(device)

    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", torch_dtype=dtype)
    target_modules = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2", "proj_out"]
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)

    lora_state_dict = load_file(lora_path)
    set_peft_model_state_dict(unet, lora_state_dict)

    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=dtype)
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

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
