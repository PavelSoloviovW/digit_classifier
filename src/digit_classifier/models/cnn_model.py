import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from digit_classifier.model import Model


class _LightningCNN(pl.LightningModule):
    """Internal PyTorch Lightning module for the Convolutional Neural Network."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 28 * 28, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNNModel(Model):
    """Convolutional Neural Network model wrapper."""

    def __init__(self):
        super().__init__()
        self.model = _LightningCNN()
        self.model.eval()

    def _reshape(self, image: np.ndarray) -> np.ndarray:
        """Ensure the 2D image has a channel dimension, resulting in 28x28x1."""
        if image.ndim == 2:
            return np.expand_dims(image, axis=-1)
        return image

    def _predict_impl(self, reshaped_image: np.ndarray) -> int:
        tensor_image = (
            torch.tensor(reshaped_image, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        with torch.no_grad():
            outputs = self.model(tensor_image)
            prediction = torch.argmax(outputs, dim=1).item()

        return int(prediction)
