from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import requests
import streamlit as st
from PIL import Image

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "mnist-8.onnx"
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx"


def download_model() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)  
    if MODEL_PATH.exists():
        return MODEL_PATH

    response = requests.get(MODEL_URL, timeout=60)
    response.raise_for_status()
    MODEL_PATH.write_bytes(response.content)
    return MODEL_PATH


@st.cache_resource
def load_session() -> ort.InferenceSession:      
    model_path = download_model()
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _center_digit(gray: np.ndarray) -> np.ndarray:  
    mask = gray > 20
    if not np.any(mask):
        return np.zeros((28, 28), dtype=np.uint8)

    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    digit = gray[y_min : y_max + 1, x_min : x_max + 1]

    h, w = digit.shape
    pad = max(h, w) // 5 + 4
    digit = cv2.copyMakeBorder(
        digit,
        pad,
        pad,
        pad,
        pad,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    h, w = digit.shape
    if h > w:
        new_h = 20
        new_w = max(1, int(round(w * 20 / h)))
    else:
        new_w = 20
        new_h = max(1, int(round(h * 20 / w)))

    resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


def preprocess_canvas_image(image_data: np.ndarray) -> tuple[np.ndarray, Image.Image]:
    rgba = image_data.astype(np.uint8)
    rgb = rgba[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    centered = _center_digit(gray)

    tensor = centered.astype(np.float32) / 255.0
    tensor = tensor.reshape(1, 1, 28, 28)

    preview = Image.fromarray(centered, mode="L")
    return tensor, preview


def predict_digit(input_tensor: np.ndarray) -> dict[str, Any]:
    session = load_session()
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    raw = outputs[0][0]
    exp = np.exp(raw - np.max(raw))
    probabilities = exp / exp.sum()

    label = int(np.argmax(probabilities))
    confidence = float(probabilities[label])

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities.tolist(),
    }
