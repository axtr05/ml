from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

X = np.array([[1,2],[2,3],[3,3],[6,7],[7,8],[8,8]])
y = np.array([0,0,0,1,1,1])

Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.4,random_state=42)

s = StandardScaler()
Xtr,Xte = s.fit_transform(Xtr), s.transform(Xte)

p = KNeighborsClassifier(3).fit(Xtr,ytr).predict(Xte)
a = accuracy_score(yte,p)
print(f'Accuracy:{a:.2f}')