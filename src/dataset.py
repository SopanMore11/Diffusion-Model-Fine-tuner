import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class StableDiffusionDataset(Dataset):
    """
    Dataset for loading local images and captions from .txt files for training.
    """
    def __init__(self, images_path: str, resolutions: list = [512, 768, 1024]):
        self.resolutions = resolutions
        self.images_path = images_path

        if not os.path.exists(images_path):
            raise FileNotFoundError(f"Images directory not found: {images_path}")

        self.image_files = []
        self.captions = []

        for file_name in os.listdir(images_path):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                image_path = os.path.join(images_path, file_name)
                caption_file = os.path.splitext(image_path)[0] + ".txt"

                if os.path.exists(caption_file):
                    self.image_files.append(file_name)
                    with open(caption_file, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                    self.captions.append(caption)
                else:
                    print(f"Warning: Caption file not found for image {file_name}")

        if len(self.image_files) == 0:
            raise ValueError("No valid image-caption pairs found!")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        print(f"Loaded {len(self.image_files)} image-caption pairs from {images_path}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        caption = self.captions[idx]
        full_image_path = os.path.join(self.images_path, image_file)

        try:
            image = Image.open(full_image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {full_image_path}: {e}")
            image = Image.new('RGB', (512, 512), color='black')

        target_res = random.choice(self.resolutions)
        width, height = image.size
        if width > height:
            new_width = target_res
            new_height = int(height * target_res / width)
        else:
            new_height = target_res
            new_width = int(width * target_res / height)

        new_width = max((new_width // 16) * 16, 256)
        new_height = max((new_height // 16) * 16, 256)

        image = image.resize((new_width, new_height), Image.LANCZOS)
        image = self.transform(image)

        return {
            'image': image,
            'caption': caption,
            'width': new_width,
            'height': new_height
        }