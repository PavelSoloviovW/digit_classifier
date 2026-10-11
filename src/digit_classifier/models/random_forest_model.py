import numpy as np
from sklearn.ensemble import RandomForestClassifier

from digit_classifier.model import Model


class RandomForestModel(Model):
    """Random Forest classifier model wrapper."""

    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        dummy_x = np.random.rand(5, 784)
        dummy_y = np.random.randint(0, 10, size=5)
        self.model.fit(dummy_x, dummy_y)

    def _predict_impl(self, reshaped_image: np.ndarray) -> int:
        prediction = self.model.predict(reshaped_image)[0]
        return int(prediction)

    def _reshape(self, image: np.ndarray) -> np.ndarray:
        """Flatten 28x28 image into a 2D array of shape (1, 784) for sklearn."""
        return image.flatten().reshape(1, -1)
