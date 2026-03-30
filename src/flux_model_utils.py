import gc
from typing import List, Sequence, Union

import bitsandbytes as bnb
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast


FLUX_LORA_TARGET_PRESETS = {
    "attention_only": [
        "to_q", "to_k", "to_v", "to_out.0",
    ],
    "attention_ff": [
        "to_q", "to_k", "to_v", "to_out.0",
        "ff.net.0.proj", "ff.net.2",
        "proj_out",
    ],
    "all_linear": [
        "to_q", "to_k", "to_v", "to_out.0",
        "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out",
        "ff.net.0.proj", "ff.net.2",
        "ff_context.net.0.proj", "ff_context.net.2",
        "proj_mlp", "proj_out",
    ],
}


def load_flux_models(
    model_name: str,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    dtype=torch.bfloat16,
    device: str = "cuda",
    lora_target_preset: str = "attention_ff",
):
    """Load all required models for FLUX training and apply LoRA to the transformer."""
    device = torch.device(device)

    print("Loading FluxTransformer2DModel")
    transformer = FluxTransformer2DModel.from_pretrained(
        model_name, subfolder="transformer", torch_dtype=dtype
    )
    transformer.enable_gradient_checkpointing()

    target_modules = FLUX_LORA_TARGET_PRESETS.get(
        lora_target_preset, FLUX_LORA_TARGET_PRESETS["attention_ff"]
    )
    print(f"Applying LoRA to FluxTransformer2DModel (targets: {lora_target_preset})")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    print("Loading AutoencoderKL (FLUX)")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=dtype)
    vae.eval()
    vae.requires_grad_(False)

    print("Loading CLIPTextModel")
    clip_text_encoder = CLIPTextModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=dtype
    )
    clip_text_encoder.eval()
    clip_text_encoder.requires_grad_(False)

    print("Loading CLIPTokenizer")
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")

    print("Loading T5EncoderModel")
    t5_text_encoder = T5EncoderModel.from_pretrained(
        model_name, subfolder="text_encoder_2", torch_dtype=dtype
    )
    t5_text_encoder.eval()
    t5_text_encoder.requires_grad_(False)

    print("Loading T5TokenizerFast")
    t5_tokenizer = T5TokenizerFast.from_pretrained(model_name, subfolder="tokenizer_2")

    print("Moving models to target device")
    transformer = transformer.to(device)
    vae = vae.to(device)
    clip_text_encoder = clip_text_encoder.to(device)
    t5_text_encoder = t5_text_encoder.to(device)

    return transformer, vae, clip_text_encoder, clip_tokenizer, t5_text_encoder, t5_tokenizer


def get_flux_scheduler(model_name: str):
    """Load FLUX flow matching scheduler."""
    print("Loading FlowMatchEulerDiscreteScheduler")
    return FlowMatchEulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")


def encode_flux_prompt(
    clip_text_encoder,
    clip_tokenizer,
    t5_text_encoder,
    t5_tokenizer,
    prompts: Union[str, List[str]],
    device,
    dtype,
    max_sequence_length: int = 512,
):
    """Encode prompts using both CLIP and T5 text encoders for FLUX."""
    if isinstance(prompts, str):
        prompts = [prompts]

    # CLIP encoding -> pooled_prompt_embeds
    clip_inputs = clip_tokenizer(
        prompts,
        max_length=clip_tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    clip_outputs = clip_text_encoder(
        input_ids=clip_inputs.input_ids.to(device),
        attention_mask=clip_inputs.attention_mask.to(device),
    )
    pooled_prompt_embeds = clip_outputs.pooler_output

    # T5 encoding -> prompt_embeds (full sequence)
    t5_inputs = t5_tokenizer(
        prompts,
        max_length=max_sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    t5_outputs = t5_text_encoder(
        input_ids=t5_inputs.input_ids.to(device),
        attention_mask=t5_inputs.attention_mask.to(device),
    )
    prompt_embeds = t5_outputs.last_hidden_state

    # text_ids are zeros for FLUX
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def pack_latents(latents, batch_size, num_channels, height, width):
    """Pack latents from [B, C, H, W] to [B, (H/2)*(W/2), C*4] for FLUX transformer."""
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
    return latents


def unpack_latents(latents, height, width, num_channels):
    """Unpack latents from [B, (H/2)*(W/2), C*4] back to [B, C, H, W]."""
    batch_size = latents.shape[0]
    latents = latents.view(batch_size, height // 2, width // 2, num_channels, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, num_channels, height, width)
    return latents


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    """Create positional IDs for latent image patches in FLUX transformer."""
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    latent_image_ids = latent_image_ids.reshape(height * width, 3)
    return latent_image_ids.to(device=device, dtype=dtype)


def flush_memory():
    """Flush GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
