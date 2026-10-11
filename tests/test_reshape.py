import numpy as np

from digit_classifier.models.cnn_model import CNNModel
from digit_classifier.models.rand_model import RandomModel
from digit_classifier.models.random_forest_model import RandomForestModel


def test_random_forest_reshape():
    """Test that RandomForestModel correctly flattens (28, 28) to (1, 784)."""
    model = RandomForestModel()
    input_image = np.zeros((28, 28), dtype=np.float32)

    reshaped = model._reshape(input_image)

    assert reshaped.shape == (1, 784)
    assert isinstance(reshaped, np.ndarray)


def test_random_model_reshape():
    """Test that RandomModel extracts a precise 10x10 center crop."""
    model = RandomModel()
    # Заполним матрицу уникальными числами для проверки правильности кропа
    input_image = np.arange(784, dtype=np.float32).reshape(28, 28)

    reshaped = model._reshape(input_image)

    assert reshaped.shape == (10, 10)
    # Проверяем, что вырезался именно центр ((28-10)//2 = 9)
    expected_crop = input_image[9:19, 9:19]
    np.testing.assert_array_equal(reshaped, expected_crop)


def test_cnn_model_reshape():
    """Test that CNNModel adds a channel dimension, resulting in (28, 28, 1)."""
    model = CNNModel()
    input_image = np.zeros((28, 28), dtype=np.float32)

    reshaped = model._reshape(input_image)

    assert reshaped.shape == (28, 28, 1)
    assert isinstance(reshaped, np.ndarray)


def test_cnn_model_reshape_already_3d():
    """Test that CNNModel handles input gracefully if channel dimension is already present."""
    model = CNNModel()
    input_image = np.zeros((28, 28, 1), dtype=np.float32)

    reshaped = model._reshape(input_image)

    assert reshaped.shape == (28, 28, 1)
