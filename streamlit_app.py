import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train import train_sd_model  # noqa: E402
from utils import load_training_config  # noqa: E402


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _normalize_uploaded_name(name: str) -> str:
    return Path(name).name.replace("\\", "_").replace("/", "_")


def _build_caption_map_from_text(text_value: str) -> Dict[str, str]:
    """Parse caption lines in format: filename|caption."""
    caption_map: Dict[str, str] = {}
    for raw_line in text_value.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "|" not in line:
            raise ValueError(f"Invalid line (missing '|'): {line}")
        filename, caption = line.split("|", 1)
        filename = filename.strip()
        caption = caption.strip()
        if not filename or not caption:
            raise ValueError(f"Invalid line (empty filename/caption): {line}")
        caption_map[Path(filename).stem] = caption
    return caption_map


def _build_caption_map_from_txt_files(caption_files) -> Dict[str, str]:
    caption_map: Dict[str, str] = {}
    for f in caption_files:
        key = Path(_normalize_uploaded_name(f.name)).stem
        caption_map[key] = f.read().decode("utf-8").strip()
    return caption_map


def _prepare_training_data(
    uploaded_images,
    caption_mode: str,
    caption_text_blob: str,
    uploaded_caption_txt,
) -> Tuple[Path, int, List[str]]:
    run_dir = ROOT_DIR / "ui_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if caption_mode == "Paste mappings (filename|caption)":
        caption_map = _build_caption_map_from_text(caption_text_blob)
    else:
        caption_map = _build_caption_map_from_txt_files(uploaded_caption_txt)

    saved_count = 0
    missing_captions: List[str] = []

    for image_file in uploaded_images:
        image_name = _normalize_uploaded_name(image_file.name)
        image_stem = Path(image_name).stem
        image_ext = Path(image_name).suffix.lower()
        if image_ext not in SUPPORTED_IMAGE_EXTENSIONS:
            continue

        image_path = images_dir / image_name
        with open(image_path, "wb") as f:
            f.write(image_file.getbuffer())

        caption = caption_map.get(image_stem, "").strip()
        if not caption:
            missing_captions.append(image_name)
            continue

        caption_path = images_dir / f"{image_stem}.txt"
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(caption)

        saved_count += 1

    return images_dir, saved_count, missing_captions


def _run_training_with_config_overrides(
    model_name: str,
    images_dir: Path,
    hf_api_key: str,
    trigger_phrase: str,
    steps: int,
    batch_size: int,
    learning_rate: float,
    sample_every: int,
    save_every: int,
):
    config_path = ROOT_DIR / "training_config.json"
    original_text = config_path.read_text(encoding="utf-8")

    try:
        config = load_training_config(str(config_path))
        config["model_name"] = model_name
        config["trigger_phrase"] = trigger_phrase
        config["steps"] = int(steps)
        config["batch_size"] = int(batch_size)
        config["learning_rate"] = float(learning_rate)
        config["sample_every"] = int(sample_every)
        config["save_every"] = int(save_every)

        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        train_sd_model(model_name=model_name, images_directory=str(images_dir), hf_api_key=hf_api_key)
    finally:
        config_path.write_text(original_text, encoding="utf-8")


st.set_page_config(page_title="Diffusion LoRA Fine-Tuner", layout="wide")
st.title("Diffusion Model Fine-Tuner (LoRA)")
st.caption("Upload image/caption pairs and launch training from one Streamlit page.")

with st.sidebar:
    st.header("Training Settings")
    default_config = load_training_config(str(ROOT_DIR / "training_config.json"))

    model_name = st.text_input("Base model name", value=default_config.get("model_name", "runwayml/stable-diffusion-v1-5"))
    hf_api_key = st.text_input("HF API key (optional)", type="password", value="")
    trigger_phrase = st.text_input("Trigger phrase", value=default_config.get("trigger_phrase", "techtron"))

    steps = st.number_input("Steps", min_value=1, value=int(default_config.get("steps", 2000)), step=1)
    batch_size = st.number_input("Batch size", min_value=1, value=int(default_config.get("batch_size", 1)), step=1)
    learning_rate = st.number_input(
        "Learning rate",
        min_value=1e-7,
        max_value=1.0,
        value=float(default_config.get("learning_rate", 1e-4)),
        format="%.7f",
    )
    sample_every = st.number_input("Sample every", min_value=1, value=int(default_config.get("sample_every", 200)), step=1)
    save_every = st.number_input("Save checkpoint every", min_value=1, value=int(default_config.get("save_every", 200)), step=1)

st.subheader("1) Upload dataset")
uploaded_images = st.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

caption_mode = st.radio(
    "Caption input mode",
    ["Paste mappings (filename|caption)", "Upload .txt caption files"],
    horizontal=True,
)

caption_text_blob = ""
uploaded_caption_txt = []
if caption_mode == "Paste mappings (filename|caption)":
    caption_text_blob = st.text_area(
        "Captions (one per line)",
        placeholder="example_1.png|A portrait of [trigger_phrase] in studio lighting",
        height=220,
    )
else:
    uploaded_caption_txt = st.file_uploader(
        "Upload caption .txt files (same basename as images)",
        type=["txt"],
        accept_multiple_files=True,
    )

st.subheader("2) Start fine-tuning")
start_training = st.button("Start Fine-tuning", type="primary", use_container_width=True)

if start_training:
    if not uploaded_images:
        st.error("Please upload at least one image.")
        st.stop()

    try:
        images_dir, saved_count, missing_captions = _prepare_training_data(
            uploaded_images=uploaded_images,
            caption_mode=caption_mode,
            caption_text_blob=caption_text_blob,
            uploaded_caption_txt=uploaded_caption_txt,
        )
    except Exception as exc:
        st.error(f"Could not prepare training data: {exc}")
        st.stop()

    if saved_count == 0:
        st.error("No valid image-caption pairs were created. Check your caption inputs.")
        if images_dir.exists():
            shutil.rmtree(images_dir.parent, ignore_errors=True)
        st.stop()

    if missing_captions:
        st.warning(f"{len(missing_captions)} image(s) were skipped due to missing captions.")
        st.write(missing_captions)

    st.success(f"Prepared {saved_count} image-caption pairs in `{images_dir}`")

    with st.spinner("Training started. This can take a long time..."):
        try:
            _run_training_with_config_overrides(
                model_name=model_name,
                images_dir=images_dir,
                hf_api_key=hf_api_key,
                trigger_phrase=trigger_phrase,
                steps=int(steps),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                sample_every=int(sample_every),
                save_every=int(save_every),
            )
            st.success("Training completed! Check the `output/` directory for weights/logs/samples.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

st.divider()
st.markdown("### Run locally")
st.code("streamlit run streamlit_app.py", language="bash")
