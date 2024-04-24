from typing import Callable, List, Tuple, Union, Dict

import numpy as np

from src.client import Client


class Server:

    def __init__(self, clients: List[Client], incentivizor: Callable[[np.ndarray], float] = lambda x: 0):
        self.server_estimations = None
        self.clients = clients
        self.incentivizor = incentivizor

    def aggregate(self) -> Dict[str, Union[np.ndarray, float]]:
        N = len(self.clients)
        local_estimations = np.zeros((N, self.clients[0].ground_truth.shape[1]))
        self.esti_var = np.zeros((N, 1))
        MSE_var = 0.

        for i in range(N):
            local_estimations[i, :] = self.clients[i].local_estimate()
            local_var = self.clients[i].bootstrap_variance()
            MSE_var += (local_var - self.clients[i].std ** 2) ** 2
            self.esti_var[i, :] = local_var

        MSE_var /= N

        self.server_estimations, estimate_W = self.algo_W(local_estimations, self.esti_var)
        self.true_estimations, W = self.true_W()
        MSE_W = np.mean((estimate_W - W) ** 2).item()
        MSE_estimations = np.mean((self.true_estimations - self.server_estimations) ** 2).item()
        return {"server_est": self.server_estimations, "MSE_var": MSE_var, "MSE_W": MSE_W, "MSE_est": MSE_estimations}

    def true_W(self) -> Tuple[np.ndarray, np.ndarray]:
        N = len(self.clients)
        self.true_var = np.zeros((N, 1))
        B = np.zeros((N, self.clients[0].ground_truth.shape[1]))
        B_opt = np.zeros((N, self.clients[0].ground_truth.shape[1]))
        for i in range(N):
            self.true_var[i, :] = self.clients[i].std ** 2
            B[i, :] = self.clients[i].local_estimate()
            B_opt[i, :] = self.clients[i].ground_truth

        V = np.eye(N) * self.true_var.squeeze(axis=1)
        C = B @ B.T
        K = B_opt @ B.T
        W = K @ np.linalg.inv(C + V)
        B = (np.expand_dims(B, axis=0).repeat(repeats=N, axis=0) * np.expand_dims(W, axis=-1)).sum(axis=1)

        return B, W

    def algo_W(self, B: np.ndarray, variances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """

        :param B: local estimators, array N * d
        :param variances: local variance estimation, array N * 1
        """
        assert B.shape[0] == variances.shape[0], f"not the same number of clients: {B.shape[0]} != {variances.shape[0]}"
        N = B.shape[0]
        V = np.eye(N) * variances.squeeze(axis=1)

        converged = False
        while not converged:
            C = B @ B.T
            K = B @ B.T
            W = K @ np.linalg.inv(C + V)
            B_update = (np.expand_dims(B, axis=0).repeat(repeats=N, axis=0) * np.expand_dims(W, axis=-1)).sum(axis=1)
            if np.allclose(B, B_update):
                converged = True
            else:
                B = B_update

        return B, W

    def gains(self) -> Tuple[np.ndarray, np.ndarray]:
        N = len(self.clients)
        client_estimate_gain = np.zeros(N)
        client_true_gain = np.zeros(N)
        for i in range(N):
            client_estimate_gain[i] = self.clients[i].gain(self.server_estimations[i])
            client_true_gain[i] = self.clients[i].gain(self.true_estimations[i])

        return client_estimate_gain, client_true_gain

