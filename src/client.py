from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class Client(ABC):

    def __init__(self, idx: int,
                 X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray | None, y_test: np.ndarray | None,
                 utility: Callable[[float], float]) -> None:
        self.idx = idx
        self.num_samples = X_train.shape[0]
        self.utility = utility

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.local = self.local_estimate()

    @abstractmethod
    def local_estimate(self) -> np.ndarray:
        raise NotImplementedError('Must be implemented in subclass.')

    def mse(self, estimate: np.ndarray) -> float:
        return np.mean((self.X_test @ estimate - self.y_test) ** 2).item()

    def gain(self, estimation_mtl: np.ndarray) -> float:
        return self.utility(self.mse(estimation_mtl)) - self.utility(self.mse(self.local))

    def bootstrap_variance(self) -> float:
        boostrap_var = []

        for _ in range(1000):
            sampling = np.random.choice(np.array(range(self.num_samples)), min(100, self.num_samples), replace=False)
            sampling = self.X_train[sampling, :]
            boostrap_var.append(np.var(sampling))

        return np.mean(boostrap_var).item()
