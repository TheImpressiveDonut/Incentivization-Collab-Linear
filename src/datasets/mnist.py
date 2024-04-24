import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

root = Path(__file__).parent.joinpath('data/mnist/root/')
dim = (28, 28)


def image_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image).reshape(-1, order="C")


def numpy_to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array.reshape(dim, order="C"))


def get_mnist() -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(root):
        os.makedirs(root)
    dataset = MNIST(root, train=True, download=True, transform=image_to_numpy)
    loader = iter(DataLoader(dataset, batch_size=len(dataset), shuffle=True))
    X, y = next(loader)
    X, y = X.numpy(), y.numpy()
    return X, y
