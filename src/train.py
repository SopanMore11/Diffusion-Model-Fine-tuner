import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm.auto import tqdm
from huggingface_hub import login as hf_login
from safetensors.torch import save_file
from peft import get_peft_model_state_dict
from diffusers import StableDiffusionPipeline, DiffusionPipeline

from dataset import StableDiffusionDataset
from load_model_utils import load_models, flush_memory, get_optimizer, get_scheduler
from utils import load_training_config, login_huggingface

login_huggingface()

def save_lora_weights(model, save_path, dtype=torch.float16):
    """
    Save LoRA weights to safetensors format.
    """
    state_dict = get_peft_model_state_dict(model)
    state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
    save_file(state_dict, save_path)
    print(f"LoRA weights saved to {save_path}")

def write_and_save_results(unet, output_dir, lora_name, training_logs, image_logs):
    """
    Save training results to local directory.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        final_save_path = os.path.join(output_dir, lora_name)
        save_lora_weights(unet, final_save_path, torch.float16)
        print(f"✅ LoRA weights saved to {final_save_path}")

        training_logs_file = os.path.join(output_dir, "training_logs.jsonl")
        with open(training_logs_file, "w") as f:
            for log in training_logs:
                f.write(json.dumps(log) + "\n")
        print(f"✅ Training logs saved to {training_logs_file}")

        image_logs_file = os.path.join(output_dir, "image_logs.jsonl")
        with open(image_logs_file, "w") as f:
            for log in image_logs:
                f.write(json.dumps(log) + "\n")
        print(f"✅ Image logs saved to {image_logs_file}")

        print(f"All training results saved locally to: {output_dir}")

    except Exception as e:
        print(f"😢 Could not save weights and logs locally: {e}")

@torch.no_grad()
def generate_samples(config, unet, vae, text_encoder, tokenizer, scheduler, prompts, output_dir, step, device, dtype, image_logs):
    """
    Generate sample images during training.
    """
    unet.eval()
    sample_dtype = torch.float32 if dtype == torch.float16 else dtype
    vae = vae.to(device=device, dtype=sample_dtype)
    text_encoder = text_encoder.to(device=device, dtype=sample_dtype)

    pipeline = StableDiffusionPipeline(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(device=device, dtype=sample_dtype)

    image_paths = []
    try:
        for i, prompt in enumerate(prompts):
            if "[trigger]" in prompt and config["trigger_phrase"]:
                prompt = prompt.replace("[trigger]", config["trigger_phrase"])
            elif config["trigger_phrase"] and config["trigger_phrase"] not in prompt:
                prompt = f"{config['trigger_phrase']} {prompt}"

            image = pipeline(
                prompt=prompt,
                width=config["sample_width"],
                height=config["sample_height"],
                num_inference_steps=config["sample_steps"],
                guidance_scale=config["guidance_scale"],
                generator=torch.Generator(device=device).manual_seed(42 + i),
            ).images[0]

            sample_dir = os.path.join(output_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            prompt_prefix = "_".join(prompt.split(" ")[:8])
            sample_path = os.path.join(sample_dir, f"step_{step}_sample_{i}_{prompt_prefix}.png")
            image.save(sample_path)
            print(f"Saved sample: {sample_path}")
            image_logs.append({"step": step, "image": os.path.basename(sample_path), "prompt": prompt, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            image_paths.append(sample_path)

    except Exception as e:
        print(f"Error generating samples: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
    finally:
        unet.train()
        vae = vae.to(device=device, dtype=dtype)
        text_encoder = text_encoder.to(device=device, dtype=dtype)
        flush_memory()

    return image_paths

def train_sd_model(model_name: str, images_directory: str, hf_api_key: str):
    """Main training function."""
    if hf_api_key:
        hf_login(hf_api_key)
    
    dtype = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_rank = 16
    lora_alpha = 16
    
    unet, vae, text_encoder, tokenizer = load_models(model_name, lora_rank, lora_alpha, dtype, device)
    
    config = load_training_config()
    config["images_path"] = images_directory

    dataset = StableDiffusionDataset(config["images_path"], resolutions=[512, 768])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    noise_scheduler = get_scheduler(model_name, dtype)
    optimizer = get_optimizer(unet, config["learning_rate"])
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    
    output_dir = os.path.join("output")
    os.makedirs(output_dir, exist_ok=True)
    config_file = os.path.join(output_dir, 'training_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    global_step = 0
    training_logs = []
    image_logs = []
    readme_lines = [f"# Stable Diffusion Fine-Tune\n\n", f"Training `{model_name}`.\n\n", "## Training Samples\n\n"]
    unet.train()
    print("Starting training...")

    while global_step < config["steps"]:
        for batch in dataloader:
            if global_step >= config["steps"]:
                break
            
            with torch.amp.autocast('cuda', dtype=dtype):
                images = batch['image'].to(device, dtype=dtype)
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                text_inputs = tokenizer(
                    batch['caption'],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                text_embeddings = text_encoder(
                    input_ids=text_inputs.input_ids.to(device),
                    attention_mask=text_inputs.attention_mask.to(device),
                )[0]
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (config['batch_size'],), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings, return_dict=False)[0]
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                loss.backward()

                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                log_entry = {
                    "step": global_step,
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                training_logs.append(log_entry)
                print(f"Step {global_step}: Loss = {loss.item():.6f}")

                if global_step % config["sample_every"] == 0:
                    print(f"Generating samples at step {global_step}")
                    image_paths = generate_samples(
                        config, unet, vae, text_encoder, tokenizer, noise_scheduler,
                        config["sample_prompts"], output_dir, global_step, device, dtype,image_logs
                    )

                    readme_lines.append(f"### Step {global_step}\n\n")
                    for image_path in image_paths:
                        readme_lines.append(f"![Sample Image]({os.path.basename(image_path)})\n\n")
                    with open(os.path.join(output_dir, "README.md"), "w") as f:
                        f.write("\n".join(readme_lines))

                if global_step % config["save_every"] == 0 and global_step > 0:
                    print(f"Saving checkpoint at step {global_step}")
                    save_path = os.path.join(output_dir, f"sd_lora_step_{global_step}.safetensors")
                    save_lora_weights(unet, save_path, torch.float16)

                global_step += 1
                flush_memory()

    print("Saving final model...")
    write_and_save_results(unet, output_dir, "sd_lora_final.safetensors", training_logs, image_logs)
    print("Training completed!")