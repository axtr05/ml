from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data = fetch_california_housing()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
plt.plot(y_pred[:50], label='Predicted', color='blue')
plt.plot(y_test[:50].values if hasattr(y_test, 'values') else y_test[:50],label='Actual',color='red')
plt.title("California Housing Price Prediction")
plt.legend()
plt.show()