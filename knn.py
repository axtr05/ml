from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X,y = load_iris(return_X_y=True)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)

s = StandardScaler()
Xtr,Xte = s.fit_transform(Xtr), s.transform(Xte)

p = KNeighborsClassifier(3).fit(Xtr,ytr).predict(Xte)
a = accuracy_score(yte,p)
print(f'Accuracy:{a:.2f}')