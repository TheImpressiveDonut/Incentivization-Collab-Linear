from typing import Callable

import numpy as np

from src.client import Client


class OLSClient(Client):

    def __init__(self, idx: int, X: np.ndarray, y: np.ndarray, utility: Callable[[float], float]):
        super().__init__(idx, X, y, utility)

    def local_estimate(self) -> np.ndarray:
        return np.linalg.pinv(self.X) @ self.y