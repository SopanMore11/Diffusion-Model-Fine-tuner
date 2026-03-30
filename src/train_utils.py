import json
import os
import re

import torch
from peft import get_peft_model_state_dict
from safetensors.torch import save_file


def save_lora_weights(model, save_path, dtype=torch.float16):
    """Save LoRA weights to safetensors format."""
    state_dict = get_peft_model_state_dict(model)
    state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
    save_file(state_dict, save_path)
    print(f"LoRA weights saved to {save_path}")


def write_and_save_results(model, output_dir, lora_name, training_logs, image_logs):
    """Save final LoRA weights and logs to local directory."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        final_save_path = os.path.join(output_dir, lora_name)
        save_lora_weights(model, final_save_path, torch.float16)
        print(f"LoRA weights saved to {final_save_path}")

        training_logs_file = os.path.join(output_dir, "training_logs.jsonl")
        with open(training_logs_file, "w", encoding="utf-8") as f:
            for log in training_logs:
                f.write(json.dumps(log) + "\n")
        print(f"Training logs saved to {training_logs_file}")

        image_logs_file = os.path.join(output_dir, "image_logs.jsonl")
        with open(image_logs_file, "w", encoding="utf-8") as f:
            for log in image_logs:
                f.write(json.dumps(log) + "\n")
        print(f"Image logs saved to {image_logs_file}")

        print(f"All training results saved locally to: {output_dir}")

    except Exception as exc:
        print(f"Could not save weights and logs locally: {exc}")


def resolve_trigger_prompt(prompt: str, trigger_phrase: str):
    """Replace trigger placeholders in prompt or prepend trigger phrase."""
    if not trigger_phrase:
        return prompt
    prompt = prompt.replace("[trigger_phrase]", trigger_phrase).replace("[trigger]", trigger_phrase)
    if trigger_phrase not in prompt:
        prompt = f"{trigger_phrase} {prompt}"
    return prompt


def sanitize_name_for_path(value: str):
    """Convert a string to a filesystem-safe name."""
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    return safe[:80] if safe else "sample"
