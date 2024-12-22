import numpy as np
from linear_regression import my_linear_regression

y = np.array([10, 50, 100, 200, 70])

X = np.array([
    [30,  7, 1, 1.50],
    [40, 10, 0, 1.60],
    [50, 20, 1, 1.70],
    [60, 12, 0, 1.80],
    [25, 10, 1, 1.65]
])

#Padronizando a base de entrada
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X_standardized = (X - mean) / std

# print(X)
# print(mean, std)
# print(X_standardized)

model = my_linear_regression()
model.fit(X_standardized, y)

prever = np.array([[40,  7, 1, 1.50]])
a = model.predict(prever)

# print(model.intercept, model.coefficients)
# print(a)
