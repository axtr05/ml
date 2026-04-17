from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

x = [[1],[2],[3],[4],[5],[6],[7],[8]]
y = [2,4,6,8,10,12,14,16]

xtr,xte,ytr,yte = train_test_split(x,y,test_size=0.25,random_state=42)

g = GridSearchCV(DecisionTreeRegressor(), {'max_depth':[1,2,3]}, cv=2).fit(xtr,ytr)
p = g.predict(xte)

print(g.best_params_)
plt.plot(p, label='Pred')
plt.plot(yte, label='Actual')
plt.legend()
plt.show()