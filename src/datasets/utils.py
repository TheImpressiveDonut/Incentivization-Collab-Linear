from typing import Tuple, List

import numpy as np

from src.datasets.mnist import get_mnist


def get_dataset(name: str, num_samples: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    if name == 'mnist':
        return get_mnist(num_samples)
    else:
        raise NotImplementedError(name)