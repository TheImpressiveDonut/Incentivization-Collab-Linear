import numpy as np

from src.client import Client


class OLSClient(Client):


    def local_estimate(self) -> np.ndarray:
        pass

    def sample(self) -> np.ndarray:
        pass