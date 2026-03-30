import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from flux_model_utils import FLUX_LORA_TARGET_PRESETS


def create_flux_inference_pipeline(
    model_name: str,
    lora_path: str,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    dtype=torch.bfloat16,
    device: str = "cuda",
    lora_target_preset: str = "attention_ff",
):
    """Load FLUX models, apply LoRA weights, and return a ready-to-use FluxPipeline."""
    device = torch.device(device)

    transformer = FluxTransformer2DModel.from_pretrained(
        model_name, subfolder="transformer", torch_dtype=dtype
    )
    target_modules = FLUX_LORA_TARGET_PRESETS.get(
        lora_target_preset, FLUX_LORA_TARGET_PRESETS["attention_ff"]
    )
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)

    lora_state_dict = load_file(lora_path)
    set_peft_model_state_dict(transformer, lora_state_dict)

    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=dtype)
    clip_text_encoder = CLIPTextModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=dtype
    )
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    t5_text_encoder = T5EncoderModel.from_pretrained(
        model_name, subfolder="text_encoder_2", torch_dtype=dtype
    )
    t5_tokenizer = T5TokenizerFast.from_pretrained(model_name, subfolder="tokenizer_2")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_name, subfolder="scheduler"
    )

    pipeline = FluxPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=clip_text_encoder,
        text_encoder_2=t5_text_encoder,
        tokenizer=clip_tokenizer,
        tokenizer_2=t5_tokenizer,
        scheduler=scheduler,
    )
    return pipeline.to(device=device, dtype=dtype)
