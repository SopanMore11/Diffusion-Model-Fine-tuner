import os
import random
from typing import List, Sequence, Tuple

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


class StableDiffusionDataset(Dataset):
    """Dataset for loading local image-caption pairs (.txt sidecars)."""

    def __init__(self, images_path: str, resolutions: Sequence[int] = (512, 768, 1024)):
        self.images_path = images_path
        self.resolutions = tuple(sorted(set(int(r) for r in resolutions)))

        if not os.path.exists(images_path):
            raise FileNotFoundError(f"Images directory not found: {images_path}")

        self.samples: List[Tuple[str, str]] = []
        for file_name in sorted(os.listdir(images_path)):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            image_path = os.path.join(images_path, file_name)
            caption_file = os.path.splitext(image_path)[0] + ".txt"
            if not os.path.exists(caption_file):
                print(f"Warning: Caption file not found for image {file_name}")
                continue

            with open(caption_file, "r", encoding="utf-8") as f:
                caption = f.read().strip()
            self.samples.append((file_name, caption))

        if not self.samples:
            raise ValueError("No valid image-caption pairs found!")

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        print(f"Loaded {len(self.samples)} image-caption pairs from {images_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_file, caption = self.samples[idx]
        full_image_path = os.path.join(self.images_path, image_file)

        try:
            with Image.open(full_image_path) as pil_img:
                image = ImageOps.exif_transpose(pil_img).convert("RGB")
        except Exception as exc:
            print(f"Error loading image {full_image_path}: {exc}")
            image = Image.new("RGB", (512, 512), color="black")

        target_res = random.choice(self.resolutions)
        width, height = image.size

        scale = target_res / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        new_width = max((new_width // 16) * 16, 256)
        new_height = max((new_height // 16) * 16, 256)

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        image = self.transform(image)

        return {
            "image": image,
            "caption": caption,
            "width": new_width,
            "height": new_height,
        }
