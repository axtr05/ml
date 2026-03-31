from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

X,y = load_diabetes(return_X_y=True)
y = (y > np.median(y)).astype(int)

X = StandardScaler().fit_transform(X)
x1,x2,y1,y2 = train_test_split(X,y,test_size=0.2,random_state=42)

p = LogisticRegression(max_iter=200).fit(x1,y1).predict(x2)

a = accuracy_score(y2,p)
print(f'Accuracy:{a:.2f}')
print("Confusion Matrix:\n", confusion_matrix(y2,p))
print("Report:\n", classification_report(y2,p))