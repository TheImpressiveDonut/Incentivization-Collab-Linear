from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class Client(ABC):

    def __init__(self, idx: int, X: np.ndarray, y: np.ndarray, utility: Callable[[float], float]) -> None:
        self.idx = idx
        self.num_samples = X.shape[0]
        self.utility = utility

        self.X = X
        self.y = y

    @abstractmethod
    def local_estimate(self) -> np.ndarray:
        pass

    def mse(self, estimation: np.ndarray) -> float:
        return np.mean((estimation - self.y) ** 2).item()

    def gain(self, estimation_mtl: np.ndarray) -> float:
        return self.utility(self.mse(estimation_mtl)) - self.utility(self.mse(self.local_estimate()))

    def bootstrap_variance(self, alpha: float = 0.95) -> float:
        boostrap_var = []

        for _ in range(1000):
            sampling = np.random.choice(np.array(range(self.num_samples)), min(100, self.num_samples), replace=False)
            sampling = self.X[sampling, :]
            boostrap_var.append(np.var(sampling))

        return np.mean(boostrap_var).item()
