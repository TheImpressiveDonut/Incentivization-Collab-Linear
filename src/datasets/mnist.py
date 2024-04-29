import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
from sklearn.datasets import fetch_openml

root = Path(__file__).parent.joinpath('data/mnist/root/')
NUM_CLASSES = 10
DIM = (28, 28)



def get_mnist(num_samples: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    if not os.path.exists(root):
        os.makedirs(root)
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    y = y.astype(np.uint8)
    X = X / 255.0

    N = num_samples.shape[0]
    num_samples = num_samples.astype(np.float32)
    num_samples /= 0.7
    data = []
    for i in range(N):
        K = np.random.choice(np.arange(10), 3, replace=False)
        prop = np.random.dirichlet(np.repeat(1.0, len(K)))
        samples = []
        for j, k in enumerate(K):
            idx = np.arange(X.shape[0])[y == k]
            samples.append(np.random.choice(idx, int(np.ceil(num_samples[i] * prop[j]).item()), replace=False))
        samples = np.concatenate(samples)
        data.append((X[samples, :], y[samples]))

    return data