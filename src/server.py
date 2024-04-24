from typing import List

import numpy as np

from src.client import Client


class Server:

    def __init__(self, clients: List[Client]):
        self.clients = clients
        self.N = len(clients)
        self.vars = np.empty(self.N)
        self.local = np.empty((self.N, self.clients[0].X.shape[1]))
        self.mtl = None
        for i in range(self.N):
            self.vars[i] = clients[i].bootstrap_variance()
            self.local[i, :] = clients[i].local_estimate()

    def aggregate(self, unbiased: bool) -> None:
        if unbiased:
            V = np.eye(self.N) * self.vars
            C = self.local @ self.local.T
            W = C @ np.linalg.inv(C + V)
            self.mtl = (np.expand_dims(self.local, axis=0).repeat(repeats=self.N, axis=0) * np.expand_dims(W, axis=-1)).sum(axis=1)
        else:
            raise NotImplementedError
