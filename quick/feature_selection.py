import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split

from quick.utils import get_mse

# Load data from https://www.openml.org/d/554

print(f'MSE before: {get_mse()}')

rr = Ridge(solver='svd')
sfs = RFECV(rr, scoring=make_scorer(mean_squared_error), n_jobs=-1)
mask = sfs.get_support()
transform_X = lambda X: X[:, mask]

print(f'MSE after: {get_mse(transform_X=transform_X)}')


