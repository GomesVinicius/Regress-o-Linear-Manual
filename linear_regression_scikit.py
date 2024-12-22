from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

y = np.array([10, 50, 100, 200, 70, 20, 19])

X = np.array([
    [30,  7, 1, 1.50, 2],
    [40, 10, 0, 1.60, 4],
    [50, 20, 1, 1.70, 5],
    [60, 12, 0, 1.80, 7],
    [25, 10, 1, 1.65, 7],
    [24, 12, 0, 1.62, 0],
    [32, 22, 1, 1.90, 0]
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

model.predict(X_train)

#Overfiting
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))
