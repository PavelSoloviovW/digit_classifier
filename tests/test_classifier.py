from unittest.mock import MagicMock

import numpy as np
import pytest

from digit_classifier.classifier import DigitClassifier
from digit_classifier.model import Model


@pytest.mark.parametrize("algo", ["cnn", "rf", "rand", "CNN"])
def test_init_with_valid_algorithms(algo):
    """Test initializing facade with valid algorithms (case-insensitive)."""
    classifier = DigitClassifier(algorithm=algo)
    assert classifier._model is not None


def test_init_with_invalid_algorithm():
    """Test that an unsupported algorithm raises ValueError."""
    with pytest.raises(ValueError):
        DigitClassifier(algorithm="unknown_algo")


def test_init_with_valid_model_injection():
    """Test direct dependency injection of a valid Model instance."""
    mock_model = MagicMock(spec=Model)
    classifier = DigitClassifier(model=mock_model)
    assert classifier._model == mock_model


def test_init_with_invalid_model_type():
    """Test that passing a non-Model object to 'model' raises TypeError."""
    with pytest.raises(TypeError):
        DigitClassifier(model="not_a_model_instance")  # type: ignore


def test_init_missing_all_arguments():
    """Test that omitting both algorithm and model raises ValueError."""
    with pytest.raises(ValueError):
        DigitClassifier()


def test_predict_delegation():
    """Test that predict method correctly delegates to the underlying model."""
    mock_model = MagicMock(spec=Model)
    mock_model.predict.return_value = 5

    classifier = DigitClassifier(model=mock_model)
    dummy_image = np.zeros((28, 28), dtype=np.float32)

    result = classifier.predict(dummy_image)

    mock_model.predict.assert_called_once_with(dummy_image)
    assert result == 5


def test_train_delegation():
    """Test that train method properly delegates to the underlying model."""
    mock_model = MagicMock(spec=Model)
    mock_model.train.side_effect = NotImplementedError("Not supported")

    classifier = DigitClassifier(model=mock_model)

    with pytest.raises(NotImplementedError):
        classifier.train()

    mock_model.train.assert_called_once()
