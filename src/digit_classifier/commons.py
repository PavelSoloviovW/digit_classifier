import numpy as np
from PIL import Image


def load_image(path: str) -> Image.Image:
    """I/O operation: loads an image from disk. (Integration scope)"""
    return Image.open(path).convert("L")


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Normalize PIL image to numpy array and validate shape."""
    img_array = np.array(image, dtype=np.float32)
    img_normalized = img_array / 255.0

    return img_normalized


def load_and_preprocess_image(path: str) -> np.ndarray:
    """Facade for I/O + processing combination."""
    img = load_image(path)
    return preprocess_image(img)
