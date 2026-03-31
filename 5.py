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

X, y = data.drop('Price',axis=1), data['Price']
    
print("Features (X):\n", X)
print("\nTarget (Y):\n", y)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
model = LinearRegression().fit(Xtr,ytr)
p = model.predict(Xte)
print("\nMean Squared Error:", mean_squared_error(yte,p))
print("Mean Absolute Error:", mean_absolute_error(yte,p))
print("\nPredicted Prices:", p)
print("Actual Prices:", yte.values)