import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
data = pd.DataFrame({
 'Area': [1200, 1500, 800, 2000, 1000],
 'Bedrooms': [3, 4, 2, 5, 2],
 'Bathrooms': [2, 3, 1, 4, 1],
 'Location': ['Downtown', 'Suburb', 'Downtown', 'Suburb', 'Uptown'],
 'Price': [300000, 400000, 200000, 500000, 250000]
})
data = pd.get_dummies(data, columns=['Location'], drop_first=True)
x = data.drop('Price', axis=1)
y = data['Price']
print("Features (X):")
print(x)
print("\nTarget (Y):")
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(x_train, y_train)
y_pred = model.predict(x_test)
print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("\nPredicted Prices:", y_pred)
print("Actual Prices:", y_test.values)
