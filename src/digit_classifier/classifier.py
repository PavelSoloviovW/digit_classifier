import numpy as np

from digit_classifier.model import Model
from digit_classifier.models import CNNModel, RandomForestModel, RandomModel


class DigitClassifier:
    """
    Thin facade class coordinating underlying models without redundant state.
    """

    def __init__(self, algorithm: str | None = None, model: Model = None):
        if model is not None:
            if not isinstance(model, Model):
                raise TypeError(
                    f"Expected an instance of Model, got {type(model).__name__}"
                )
            self._model = model
            return

        if algorithm is not None:
            self._model = self._resolve_model(algorithm)
            return

        raise ValueError("Either 'algorithm' or 'model' must be provided.")

    def predict(self, image: np.ndarray) -> int:
        """Delegates prediction directly to the resolved model."""
        return self._model.predict(image)

    def train(self, *args, **kwargs) -> None:
        """Delegates training, raising NotImplementedError."""
        return self._model.train(*args, **kwargs)

    @staticmethod
    def _resolve_model(algorithm: str) -> Model:
        """Pure static function to resolve and instantiate the model by algorithm name."""
        algo = algorithm.lower()
        if algo == "cnn":
            return CNNModel()
        elif algo == "rf":
            return RandomForestModel()
        elif algo == "rand":
            return RandomModel()
        else:
            raise ValueError(
                f"Unsupported algorithm: '{algo}'. Choose from: 'cnn', 'rf', 'rand'."
            )


DigitClassificationInterface = Model
