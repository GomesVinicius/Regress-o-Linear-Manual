import numpy as np

class my_linear_regression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.intercept = None
        self.coefficients = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        # self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

        #Gradiente Descendente
        m, n = X.shape
        self.theta = np.zeros(n)

        for x in range(self.n_iterations):
            gradients = (2/m) * X.T.dot(X.dot(self.theta) - y)
            self.theta -= self.learning_rate * gradients

        self.intercept = self.theta[0]
        self.coefficients = self.theta[1:]

    def predict(self, X):
        Xb = np.c_[ np.ones(X.shape[0]), X]

        return Xb.dot( np.r_[self.intercept, self.coefficients] )
