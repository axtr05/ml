import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.DataFrame({
 'Area':[1200,1500,800,2000,1000],
 'Bedrooms':[3,4,2,5,2],
 'Bathrooms':[2,3,1,4,1],
 'Location':['Downtown','Suburb','Downtown','Suburb','Uptown'],
 'Price':[300000,400000,200000,500000,250000]
})

data = pd.get_dummies(data, columns=['Location'], drop_first=True)
x, y = data.drop('Price', axis=1), data['Price']

xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=42)
p = LinearRegression().fit(xtr, ytr).predict(xte)

print("MSE:", mean_squared_error(yte, p))
print("MAE:", mean_absolute_error(yte, p))
print("Predicted:", p)
print("Actual:", yte.values)