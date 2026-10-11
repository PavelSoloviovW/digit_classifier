import numpy as np

from digit_classifier.model import Model


class RandomModel(Model):
    """Random prediction model that applies a 10x10 center crop."""

    def _reshape(self, image: np.ndarray) -> np.ndarray:
        """Crop the center 10x10 pixels from the 28x28 image."""
        h, w = image.shape
        start_h = (h - 10) // 2  # (28 - 10) // 2 = 9
        start_w = (w - 10) // 2  # (28 - 10) // 2 = 9
        return image[start_h : start_h + 10, start_w : start_w + 10]

    def _predict_impl(self, reshaped_image: np.ndarray) -> int:
        """Return a random digit prediction based on the cropped image."""
        assert reshaped_image.shape == (10, 10), (
            f"Expected shape (10, 10), got {reshaped_image.shape}"
        )
        return int(np.random.randint(0, 10))
