import os

import numpy as np
from PIL import Image

output_dir = "./images"
os.makedirs(output_dir, exist_ok=True)

mnist_array = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
Image.fromarray(mnist_array).save(os.path.join(output_dir, "mnist_sample.png"))

invalid_array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
Image.fromarray(invalid_array).save(os.path.join(output_dir, "non_mnist_sample.png"))

print(
    f"Test images successfully created in '{output_dir}': 'mnist_sample.png' and 'non_mnist_sample.png'"
)
