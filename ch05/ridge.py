import numpy as np
from scipy import linalg

class RidgeRegression:
    def __init__(self, lambda_=1.):
        self.lambda_ = lambda_
        self.w_ = None

    def fit(self, X, t):
        Xtill = np.c_[np.ones(X.shape[0]), X]
        c = np.eye(Xtill.shape[1])
        A = np.dot(Xtill.T, Xtill) + self.lambda_ * c
        b = np.dot(Xtill.T, t)
        self.w_ = linalg.solve(A, b)

    def predict(self, X):
        Xtil = np.c_[np.ones(X.shape[0]), X]
        return np.dot(Xtil, self.w_)

