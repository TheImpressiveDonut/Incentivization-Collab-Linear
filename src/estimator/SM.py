from typing import Callable

import numpy as np

from src.client import Client

class SMClient(Client):

    def __init__(self, id: int, num_samples: int, y: np.ndarray,
                 utility: Callable[[float], float]):
        self.std = np.random.random() * 2 + 2
        self.y = y
        X_train = np.random.normal(y.item(), self.std, (num_samples, 1))
        super().__init__(id,
                         X_train, y, None, None,
                         utility)


    def local_estimate(self) -> np.ndarray:
        return self.X_train.mean(axis=0, keepdims=True)


    def mse(self, estimate: np.ndarray) -> float:
        return np.sum((estimate - self.y) ** 2).item()