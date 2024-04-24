from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class Client(ABC):

    def __init__(self, id: int, num_samples: int, y: np.ndarray, utility: Callable[[float], float]) -> None:
        self.id = id
        self.num_samples = num_samples
        self.ground_truth = ground_truth
        self.utility = utility

        self.std = np.random.random() * 5
        self.X = self.sample()  # n * d

    @abstractmethod
    def sample(self) -> np.ndarray:
        pass

    @abstractmethod
    def local_estimate(self) -> np.ndarray:
        pass

    def mse(self, estimation: np.ndarray) -> float:
        return np.mean((estimation - self.ground_truth) ** 2).item()

    def gain(self, estimation_mtl: np.ndarray) -> float:
        return self.utility(self.mse(estimation_mtl)) - self.utility(self.mse(self.local_estimate()))

    def bootstrap_variance(self, alpha: float = 0.95) -> float:
        boostrap_var = []

        for _ in range(1000):
            sampling = np.random.choice(np.array(range(self.num_samples)), min(100, self.num_samples), replace=False)
            sampling = self.X[sampling, :]
            boostrap_var.append(np.var(sampling))

        return np.mean(boostrap_var).item()
