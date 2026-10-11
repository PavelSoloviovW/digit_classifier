from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    """Abstract Base Class implementing the Template Method pattern."""

    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError("Training is not supported for this architecture.")

    def predict(self, image: np.ndarray) -> int:
        """Template method: validates image first, then runs prediction."""
        self._validate_image(image)
        reshaped = self._reshape(image)
        return self._predict_impl(reshaped)

    @abstractmethod
    def _predict_impl(self, image: np.ndarray) -> int:
        """Core prediction logic to be implemented by subclasses."""

    def _validate_image(self, image: np.ndarray) -> None:
        """Common validation logic including shape check for standard models."""
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy.ndarray.")

        if image.shape != (28, 28):
            raise ValueError(f"Expected image shape (28, 28), got {image.shape}")

    @abstractmethod
    def _reshape(self, image: np.ndarray) -> np.ndarray:
        """Mandatory reshape/transformation logic for all concrete models."""
