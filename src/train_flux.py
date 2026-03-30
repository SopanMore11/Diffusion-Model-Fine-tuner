import json
import os
from contextlib import nullcontext
from datetime import datetime

import torch
import torch.nn.functional as F
from diffusers import FluxPipeline
from huggingface_hub import login as hf_login
from torch.utils.data import DataLoader

from dataset import StableDiffusionDataset
from flux_model_utils import (
    encode_flux_prompt,
    flush_memory,
    get_flux_scheduler,
    load_flux_models,
    pack_latents,
    prepare_latent_image_ids,
)
from load_model_utils import get_optimizer
from train_utils import resolve_trigger_prompt, sanitize_name_for_path, save_lora_weights, write_and_save_results
from utils import load_training_config


@torch.no_grad()
def generate_flux_samples(
    config,
    transformer,
    vae,
    clip_text_encoder,
    clip_tokenizer,
    t5_text_encoder,
    t5_tokenizer,
    scheduler,
    prompts,
    output_dir,
    step,
    device,
    dtype,
    image_logs,
):
    """Generate sample images during FLUX training."""
    transformer.eval()

    pipeline = FluxPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=clip_text_encoder,
        text_encoder_2=t5_text_encoder,
        tokenizer=clip_tokenizer,
        tokenizer_2=t5_tokenizer,
        scheduler=scheduler,
    )
    pipeline = pipeline.to(device=device, dtype=dtype)

    sample_dir = os.path.join(output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    image_paths = []
    try:
        for i, prompt in enumerate(prompts):
            prompt = resolve_trigger_prompt(prompt, config.get("trigger_phrase", ""))
            image = pipeline(
                prompt=prompt,
                width=config.get("flux_sample_width", 1024),
                height=config.get("flux_sample_height", 1024),
                num_inference_steps=config.get("flux_sample_steps", 4),
                guidance_scale=config.get("flux_guidance_scale", 0.0),
                max_sequence_length=config.get("flux_max_sequence_length", 512),
                generator=torch.Generator(device=device).manual_seed(42 + i),
            ).images[0]

            prompt_prefix = sanitize_name_for_path("_".join(prompt.split(" ")[:8]))
            sample_path = os.path.join(sample_dir, f"step_{step}_sample_{i}_{prompt_prefix}.png")
            image.save(sample_path)
            print(f"Saved sample: {sample_path}")
            image_logs.append(
                {
                    "step": step,
                    "image": os.path.basename(sample_path),
                    "prompt": prompt,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            image_paths.append(sample_path)
    except Exception as exc:
        print(f"Error generating FLUX samples: {exc}")
    finally:
        transformer.train()
        del pipeline
        flush_memory()

    return image_paths


def train_flux_model(model_name: str, images_directory: str, hf_api_key: str):
    """Main training function for FLUX Schnell."""
    if hf_api_key:
        hf_login(hf_api_key)

    config = load_training_config()
    config["images_path"] = images_directory

    # FLUX works best with bfloat16 on supported GPUs
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_rank = int(config.get("lora_rank", 16))
    lora_alpha = int(config.get("lora_alpha", 16))
    lora_target_preset = config.get("flux_lora_targets", "attention_ff")

    transformer, vae, clip_text_encoder, clip_tokenizer, t5_text_encoder, t5_tokenizer = (
        load_flux_models(model_name, lora_rank, lora_alpha, dtype, device, lora_target_preset)
    )

    flux_res = config.get("resolutions", [1024])
    dataset = StableDiffusionDataset(config["images_path"], resolutions=flux_res)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )

    scheduler = get_flux_scheduler(model_name)
    optimizer = get_optimizer(transformer, config["learning_rate"])
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    grad_accum_steps = max(1, int(config.get("gradient_accumulation_steps", 1)))
    max_seq_len = int(config.get("flux_max_sequence_length", 512))

    # FLUX VAE config values
    vae_scaling_factor = vae.config.scaling_factor
    vae_shift_factor = getattr(vae.config, "shift_factor", 0.0)
    num_latent_channels = vae.config.latent_channels

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    config_file = os.path.join(output_dir, "training_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    global_step = 0
    training_logs = []
    image_logs = []
    readme_lines = [
        "# FLUX Schnell Fine-Tune\n\n",
        f"Training `{model_name}`.\n\n",
        "## Training Samples\n\n",
    ]
    transformer.train()
    optimizer.zero_grad(set_to_none=True)
    print("Starting FLUX training...")

    while global_step < config["steps"]:
        for batch in dataloader:
            if global_step >= config["steps"]:
                break

            autocast_ctx = (
                torch.amp.autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext()
            )
            with autocast_ctx:
                images = batch["image"].to(device, dtype=dtype, non_blocking=True)
                batch_size_actual = images.shape[0]

                with torch.no_grad():
                    # 1. Encode images to latents
                    latents = vae.encode(images).latent_dist.sample()
                    latents = (latents - vae_shift_factor) * vae_scaling_factor

                    _, _, latent_h, latent_w = latents.shape

                    # 2. Pack latents for FLUX transformer (2x2 patches)
                    packed_latents = pack_latents(
                        latents, batch_size_actual, num_latent_channels, latent_h, latent_w
                    )

                    # 3. Encode text with dual encoders
                    prompt_embeds, pooled_prompt_embeds, text_ids = encode_flux_prompt(
                        clip_text_encoder,
                        clip_tokenizer,
                        t5_text_encoder,
                        t5_tokenizer,
                        batch["caption"],
                        device,
                        dtype,
                        max_seq_len,
                    )

                    # 4. Prepare latent image IDs (positional info)
                    latent_image_ids = prepare_latent_image_ids(
                        batch_size_actual, latent_h // 2, latent_w // 2, device, dtype
                    )

                # 5. Sample timesteps (sigmoid sampling for flow matching)
                u = torch.sigmoid(torch.randn((batch_size_actual,), device=device, dtype=dtype))
                timesteps = u * 1000.0

                # 6. Create noisy latents using rectified flow ODE
                noise = torch.randn_like(packed_latents)
                t_expanded = u.view(-1, 1, 1)
                noisy_latents = (1.0 - t_expanded) * packed_latents + t_expanded * noise

                # 7. Guidance embedding (0.0 for Schnell)
                guidance = torch.full(
                    (batch_size_actual,),
                    config.get("flux_guidance_scale", 0.0),
                    device=device,
                    dtype=dtype,
                )

                # 8. Forward pass through FLUX transformer
                noise_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps / 1000.0,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                # 9. Flow matching loss: target is velocity (noise - clean_latents)
                target = noise - packed_latents
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                (loss / grad_accum_steps).backward()

            should_step = (global_step + 1) % grad_accum_steps == 0
            if should_step:
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            log_entry = {
                "step": global_step,
                "loss": float(loss.item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            training_logs.append(log_entry)
            print(f"Step {global_step}: Loss = {loss.item():.6f}")

            if global_step % config["sample_every"] == 0:
                print(f"Generating FLUX samples at step {global_step}")
                image_paths = generate_flux_samples(
                    config,
                    transformer,
                    vae,
                    clip_text_encoder,
                    clip_tokenizer,
                    t5_text_encoder,
                    t5_tokenizer,
                    scheduler,
                    config["sample_prompts"],
                    output_dir,
                    global_step,
                    device,
                    dtype,
                    image_logs,
                )

                readme_lines.append(f"### Step {global_step}\n\n")
                for image_path in image_paths:
                    rel_path = os.path.relpath(image_path, output_dir)
                    readme_lines.append(f"![Sample Image]({rel_path})\n\n")
                with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
                    f.write("\n".join(readme_lines))

            if global_step % config["save_every"] == 0 and global_step > 0:
                print(f"Saving FLUX checkpoint at step {global_step}")
                save_path = os.path.join(output_dir, f"flux_lora_step_{global_step}.safetensors")
                save_lora_weights(transformer, save_path, torch.float16)

            global_step += 1

    print("Saving final FLUX model...")
    write_and_save_results(transformer, output_dir, "flux_lora_final.safetensors", training_logs, image_logs)
    print("FLUX training completed!")
