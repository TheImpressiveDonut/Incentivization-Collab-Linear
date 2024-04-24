from typing import Callable

import numpy as np

from src.client import Client

class ScalarMeanClient(Client):

    def __init__(self, id: int, num_samples: int, ground_truth: np.ndarray,
                 utility: Callable[[float], float]):
        super().__init__(id, num_samples, ground_truth, utility)

    def sample(self) -> np.ndarray:
        self.std = np.random.random() * 5
        return np.random.normal(self.y.item(), self.std, (self.num_samples, 1))

    def local_estimate(self) -> np.ndarray:
        return self.X.mean(axis=0, keepdims=True)