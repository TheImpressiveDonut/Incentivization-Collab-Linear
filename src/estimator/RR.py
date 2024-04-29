from typing import Callable, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from src.client import Client


class RRClient(Client):

    def __init__(self, idx: int, data: Tuple[np.ndarray, np.ndarray], alpha: float,
                 utility: Callable[[float], float]):
        X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.7)
        self.alpha = alpha
        super().__init__(idx,
                         X_train, y_train, X_test, y_test,
                         utility)

    def local_estimate(self) -> np.ndarray:
        rr = Ridge(alpha=self.alpha, solver='svd', fit_intercept=False)
        rr.fit(self.X_train, self.y_train)
        return rr.coef_
