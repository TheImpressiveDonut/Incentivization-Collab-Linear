from typing import Callable, Any, List

import numpy as np


class Server:

    def __init__(self, incentivizor: Callable[[np.ndarray], float]):
        self.incentivizor = incentivizor

    def aggregate(self, local_estimates: np.ndarray, local_variances: np.ndarray) -> np.ndarray:
        return self.algo_W(local_estimates, local_variances)


    def algo_W(self, B: np.ndarray, variances: np.ndarray) -> np.ndarray:
        """

        :param B: local estimators, array N * d
        :param variances: local variance estimation, array N * 1
        """
        assert B.shape[0] == variances.shape[0], f"not the same number of clients: {B.shape[0]} != {variances.shape[0]}"
        N = B.shape[0]
        V = np.eye(N) * variances.squeeze(axis=1)
        while True:
            C = B @ B.T
            K = B @ B.T
            W = K @ np.linalg.inv(C + V)
            B = (np.expand_dims(B, axis=0).repeat(repeats=N, axis=0) * np.expand_dims(W, axis=-1)).sum(axis=1)
            break

        return B