from typing import Callable

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def get_mse(
        transform_X: Callable[[np.ndarray], np.ndarray] = lambda X: X,
        transform_y: Callable[[np.ndarray], np.ndarray] = lambda y: y,
)-> float:
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    y = y.astype(np.uint8)
    X = X / 255.
    X = transform_X(X)
    y = transform_y(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.7)

    rr = Ridge(solver='svd')
    rr.fit(X_train, y_train)

    y_pred = rr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse