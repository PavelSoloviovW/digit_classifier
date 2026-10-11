import numpy as np
import pytest

from digit_classifier.model import Model


class DummyModel(Model):
    """Concrete dummy model for testing base class methods."""

    def _predict_impl(self, image: np.ndarray) -> int:
        return 0

    def _reshape(self, image: np.ndarray) -> np.ndarray:
        return image

    def predict(self, image: np.ndarray) -> int:
        self._validate_image(image)
        return self._predict_impl(image)


def test_validate_image_valid():
    """Test that _validate_image successfully accepts a standard 28x28 image."""
    model = DummyModel()
    valid_image = np.zeros((28, 28), dtype=np.float32)
    model._validate_image(valid_image)


@pytest.mark.parametrize(
    "invalid_shape",
    [
        (28, 27),
        (32, 32),
        (28, 28, 3),
        (784,),
        (100, 100, 3),
    ],
)
def test_validate_image_invalid(invalid_shape):
    """Test that _validate_image raises ValueError for incorrect image shapes."""
    model = DummyModel()
    invalid_image = np.zeros(invalid_shape, dtype=np.float32)
    with pytest.raises(ValueError):
        model._validate_image(invalid_image)
