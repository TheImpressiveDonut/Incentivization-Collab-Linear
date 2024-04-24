from typing import Tuple

import numpy as np

from src.datasets.mnist import get_mnist


def get_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    if name == 'mnist':
        return get_mnist()
    else:
        raise NotImplementedError(name)