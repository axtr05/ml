import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
np.random.seed(42)
X = 2*np.random.rand(100,1)
Y = 4 + 3*X + np.random.randn(100,1)
model = LinearRegression().fit(X, Y)
print(model.intercept_[0], model.coef_[0][0])
plt.scatter(X, Y)
plt.plot(X, model.predict(X), 'r')
plt.show()  