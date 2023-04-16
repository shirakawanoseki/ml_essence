
import numpy as np
from scipy import linalg

class LinearRegression:
    def __init__(self) -> None:
        self.w_ = None

    def fit(self, X, t):
        Xtill = np.c_[np.ones(X.shape[0]), X]
        A = np.dot(Xtill.T, Xtill)
        b = np.dot(Xtill.T, t)
        self.w_ = linalg.solve(A, b)

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xtil = np.c_[np.ones(X.shape[0]), X]
        return np.dot(Xtil, self.w_)
