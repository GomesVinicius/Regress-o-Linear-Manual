import numpy as np

class my_linear_regression:
    def __init__(self):
        self.alfa = None
        self.betas = None

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        betas = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

        self.alfa = betas[0]
        self.betas = betas[1:]

    def predict(self, X):
        Xb = np.c_[ np.ones(X.shape[0]), X]

        return Xb.dot( np.r_[self.alfa, self.betas] )
