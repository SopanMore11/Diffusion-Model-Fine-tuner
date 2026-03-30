# LoRA Fine-Tuning for Diffusion Models

A modular pipeline for fine-tuning **Stable Diffusion** and **FLUX.1 Schnell** models using Low-Rank Adaptation (LoRA). It provides a clear, maintainable, and configurable training and inference system with a Streamlit UI.

---

## Supported Models

| Model | Architecture | Default Resolution | Inference Steps |
|-------|-------------|-------------------|-----------------|
| **Stable Diffusion v1.5** | UNet + CLIP | 512px | 20 |
| **FLUX.1 Schnell** | Transformer + CLIP + T5 | 1024px | 1-4 |

---

## Features

* **Multi-model support**: Train on Stable Diffusion or FLUX Schnell with a single `model_type` switch.
* **Modular codebase**: Separate training files for each model (`train_sd.py`, `train_flux.py`) with shared utilities.
* **LoRA fine-tuning**: Efficiently train only a small number of parameters, reducing GPU memory and training time.
* **Configurable training**: All parameters controlled via `training_config.json`.
* **Automatic sample generation**: Generate sample images at regular intervals to monitor progress.
* **`safetensors` support**: LoRA weights saved in the recommended format.
* **Streamlit UI**: Upload datasets and launch training from a single web page.

---

## Project Structure

```
src/
├── train.py              # Dispatcher — routes to SD or FLUX trainer
├── train_sd.py           # Stable Diffusion training loop + sample generation
├── train_flux.py         # FLUX Schnell training loop + sample generation
├── train_utils.py        # Shared utilities (saving weights, logs, prompt handling)
├── load_model_utils.py   # SD model loading (UNet, VAE, CLIP) + optimizer
├── flux_model_utils.py   # FLUX model loading (Transformer, VAE, CLIP, T5) + latent helpers
├── inference.py          # SD inference pipeline
├── flux_inference.py     # FLUX inference pipeline
├── dataset.py            # Image-caption dataset (shared by both models)
└── utils.py              # Config loading, HF login
```

---

## Getting Started

### Prerequisites

* Python 3.8+
* **Stable Diffusion**: CUDA GPU with 8GB+ VRAM (16GB recommended)
* **FLUX Schnell**: CUDA GPU with 24GB+ VRAM (due to dual text encoders + transformer)
* Hugging Face account (for gated model downloads)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your_username/Diffusion-Model-Fine-tuner.git
    cd Diffusion-Model-Fine-tuner
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Prepare your dataset. Place images in a directory with matching `.txt` caption files:

    ```
    /your_images_path/
    ├── image1.png
    ├── image1.txt    # "A photo of [trigger_phrase] playing guitar"
    ├── image2.png
    └── image2.txt    # "An illustration of [trigger_phrase] reading a book"
    ```

---

## Configuration

The `training_config.json` file controls all training parameters.

### Stable Diffusion

```json
{
  "model_type": "sd",
  "model_name": "runwayml/stable-diffusion-v1-5",
  "images_path": "path/to/your/images",
  "trigger_phrase": "techtron",
  "batch_size": 1,
  "steps": 2000,
  "learning_rate": 1e-4,
  "lora_rank": 16,
  "lora_alpha": 16,
  "sample_width": 512,
  "sample_height": 512,
  "guidance_scale": 7.5,
  "sample_steps": 20
}
```

### FLUX Schnell

Set `model_type` to `"flux"` and configure the FLUX-specific fields:

```json
{
  "model_type": "flux",
  "model_name": "black-forest-labs/FLUX.1-schnell",
  "images_path": "path/to/your/images",
  "trigger_phrase": "techtron",
  "batch_size": 1,
  "steps": 1000,
  "learning_rate": 1e-4,
  "lora_rank": 16,
  "lora_alpha": 16,
  "flux_sample_steps": 4,
  "flux_sample_width": 1024,
  "flux_sample_height": 1024,
  "flux_guidance_scale": 0.0,
  "flux_max_sequence_length": 512,
  "flux_lora_targets": "attention_ff"
}
```

#### FLUX LoRA Target Presets

| Preset | Description |
|--------|-------------|
| `attention_only` | Only attention layers (to_q, to_k, to_v, to_out) |
| `attention_ff` | Attention + feed-forward layers (recommended) |
| `all_linear` | All linear layers including cross-attention and context FF |

---

## Usage

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

1. Select **Model type** (`sd` or `flux`) in the sidebar.
2. Configure training parameters (steps, learning rate, batch size, etc.).
3. Upload images and captions.
4. Click **Start Fine-tuning**.

FLUX-specific settings (sample steps, resolution, LoRA targets) appear automatically when `flux` is selected.

### Python API

```python
import sys
sys.path.insert(0, "src")
from train import train_model

# Stable Diffusion
train_model(
    model_name="runwayml/stable-diffusion-v1-5",
    images_directory="path/to/images",
    hf_api_key="",
    model_type="sd",
)

# FLUX Schnell
train_model(
    model_name="black-forest-labs/FLUX.1-schnell",
    images_directory="path/to/images",
    hf_api_key="",
    model_type="flux",
)
```

### Inference

```python
# Stable Diffusion
from inference import create_inference_pipeline

pipe = create_inference_pipeline(
    model_name="runwayml/stable-diffusion-v1-5",
    lora_path="output/sd_lora_final.safetensors",
)
image = pipe("A photo of techtron in a park").images[0]

# FLUX Schnell
from flux_inference import create_flux_inference_pipeline

pipe = create_flux_inference_pipeline(
    model_name="black-forest-labs/FLUX.1-schnell",
    lora_path="output/flux_lora_final.safetensors",
)
image = pipe("A photo of techtron in a park", num_inference_steps=4).images[0]
```

---

## Outputs

Training produces the following in the `output/` directory:

```
output/
├── sd_lora_final.safetensors      # or flux_lora_final.safetensors
├── *_lora_step_*.safetensors      # Intermediate checkpoints
├── training_config.json           # Config snapshot for the run
├── training_logs.jsonl            # Per-step loss and learning rate
├── image_logs.jsonl               # Sample generation metadata
├── samples/                       # Generated sample images
└── README.md                      # Auto-generated with embedded samples
```

---

## Key Technical Details

### Stable Diffusion
- **Backbone**: UNet2DConditionModel with LoRA on attention + FF layers
- **Text encoder**: CLIP (frozen)
- **Loss**: MSE on predicted noise vs actual noise (DDPM)
- **dtype**: float16

### FLUX Schnell
- **Backbone**: FluxTransformer2DModel with LoRA
- **Text encoders**: CLIP (pooled embeddings) + T5 (sequence embeddings), both frozen
- **Loss**: MSE on predicted velocity vs true velocity (rectified flow matching)
- **Latent format**: 16-channel VAE latents packed into 2x2 patches
- **Timestep sampling**: Sigmoid distribution (concentrates on mid-range timesteps)
- **dtype**: bfloat16 (required — float16 overflows for FLUX)
- **Guidance**: 0.0 (Schnell is a distilled model, no classifier-free guidance)
