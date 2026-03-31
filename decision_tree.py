from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

X,y = fetch_california_housing(return_X_y=True)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)

g = GridSearchCV(DecisionTreeRegressor(), {'max_depth':[2,4,6,8]}, cv=3).fit(Xtr,ytr)
p = g.predict(Xte)

print(g.best_params_)

plt.plot(p[:50], label='Predicted')
plt.plot(yte[:50], label='Actual')
plt.title("Decision Tree Prediction")
plt.legend()
plt.show()