from typing import List, Callable

import numpy as np

from src.client import Client


class Server:

    def __init__(self, clients: List[Client]):
        self.clients = clients
        self.N = len(clients)
        self.vars = np.empty(self.N)
        self.local = np.empty((self.N, self.clients[0].X_train.shape[1]))
        self.mtl = None
        for i in range(self.N):
            self.vars[i] = clients[i].bootstrap_variance()
            self.local[i, :] = clients[i].local

    def aggregate(self, unbiased: bool,
                  inc: Callable[[int, np.ndarray], float] = lambda x, y: 0
                  ) -> np.ndarray:
        if unbiased:
            V = np.eye(self.N) * self.vars
            C = self.local @ self.local.T
            W = C @ np.linalg.inv(C + V)

            self.mtl = np.empty((self.N, self.clients[0].X_train.shape[1]))
            for i in range(self.N):
                self.mtl[i, :] = np.sum(self.local * W[i, :].reshape(self.N, 1), axis=0) + inc(i, W) * self.vars[i]
            return W
        else:
            raise NotImplementedError
