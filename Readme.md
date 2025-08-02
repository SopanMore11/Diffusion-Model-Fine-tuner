# LoRA Fine-Tuning for Stable Diffusion: A Portfolio Project

This project showcases a professional and modular pipeline for fine-tuning a Stable Diffusion model using Low-Rank Adaptation (LoRA). It demonstrates practical skills in deep learning, MLOps principles, and Python development by creating a clear, maintainable, and highly configurable training and inference system.

The goal of this project is to provide a robust framework for creating custom Stable Diffusion models for a specific subject or style with minimal computational overhead. The entire process, from data preparation to model inference, is designed to be easily reproducible and scalable.

---

## ✨ Features

* **Modular Codebase**: The project is structured into logical Python files (`model_utils.py`, `dataset.py`, `training_and_sampling.py`, `inference.py`), making it easy to understand, modify, and extend.
* **LoRA Support**: Efficiently fine-tune large Stable Diffusion models by only training a small number of parameters, significantly reducing GPU memory usage and training time.
* **Configurable Training**: All training parameters are defined in an external `training_config.json` file, allowing for easy experimentation and version control of different training runs.
* **Automatic Sample Generation**: Generate sample images at regular intervals during training to monitor progress and provide visual feedback.
* **`safetensors` support**: The final LoRA weights are saved in the recommended `safetensors` format for secure and fast loading.

---

## 🚀 Getting Started

### Prerequisites

You will need the following to run this project:

* Python 3.8 or higher
* A CUDA-compatible GPU with at least 8GB of VRAM (16GB recommended for larger resolutions)
* Hugging Face account for model downloads (if using gated models)

### Installation

1.  Clone the repository and navigate to the project directory.

    ```bash
    git clone [https://github.com/your_username/sd-lora-finetune.git](https://github.com/your_username/sd-lora-finetune.git)
    cd sd-lora-finetune
    ```

2.  Prepare your dataset. Place your images in a directory, with each image having a corresponding `.txt` file containing its caption. The caption should include your unique `trigger_phrase`.

    **Example:**

    ```
    /your_images_path/
    ├── my_image1.png
    ├── my_image1.txt  (e.g., "A photo of [trigger] playing guitar")
    ├── my_image2.png
    └── my_image2.txt  (e.g., "An illustration of [trigger] reading a book")
    ```

---

## ⚙️ Configuration

The `training_config.json` file controls all aspects of the training process. Update the values in this file to match your project.

```json
{
  "model_name": "runwayml/stable-diffusion-v1-5",
  "images_path": "path/to/your/images",
  "hf_api_key": null,
  "trigger_phrase": "techtron",
  "batch_size": 1,
  "gradient_accumulation_steps": 1,
  "steps": 2000,
  "learning_rate": 1e-4,
  "optimizer": "adamw8bit",
  "lora_rank": 16,
  "lora_alpha": 16,
  "save_every": 200,
  "sample_every": 200,
  "max_step_saves": 4,
  "save_dtype": "float16",
  "sample_width": 512,
  "sample_height": 512,
  "guidance_scale": 7.5,
  "sample_steps": 20,
  "sample_prompts": [
    "[trigger] linkedin headshot, professional, high quality, studio lighting, corporate background, sharp focus, dslr photo",
    "[trigger] instagram post, vibrant colors, influencer style, street fashion, bokeh background, dramatic lighting",
    "[trigger] facebook profile picture, friendly, casual, outdoors, smiling, natural light, park setting",
    "[trigger] fashion model for a magazine cover, dramatic pose, high fashion attire, bold colors, magazine logo",
    "[trigger] video game character, detailed armor, futuristic city background, neon lights, high resolution, concept art",
    "[trigger] professional chef, elegant, cooking in a modern kitchen, gourmet dish, steam rising, cinematic",
    "[trigger] professional photographer, holding a camera, urban exploration, golden hour, wide shot",
    "[trigger] musician, playing a guitar on stage, concert lighting, energetic crowd, rock star, detailed",
    "[trigger] painter in a studio, artistic style, easel, brushes, canvas, natural light, soft colors",
    "[trigger] professional athlete, action shot, dynamic pose, sports stadium, bright sunlight, motion blur"
  ]
}