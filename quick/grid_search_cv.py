import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV

from quick.utils import get_mse

print(f'MSE before: {get_mse()}')

X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
y = y.astype(np.uint8)
X = X / 255.

rr = Ridge(solver='svd')
gs = GridSearchCV(rr, scoring=make_scorer(mean_squared_error), param_grid={'alpha': np.logspace(-3, 1, 4)}, n_jobs=-1)
gs.fit(X, y)

print(f'MSE after: {get_mse(alpha=gs.best_params_["alpha"])}')

