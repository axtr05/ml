from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

x = np.array([[1,2],[9,2],[2,3],[3,4],[5,5],[4,6],[6,7],[9,7]])
y = np.array([0,0,0,1, 0,1,1,1])

x = StandardScaler().fit_transform(x)
x1,x2,y1,y2 = train_test_split(x,y,test_size=0.33,random_state=42)

p = LogisticRegression().fit(x1,y1).predict(x2)

print(f'Accuracy:{accuracy_score(y2,p):.2f}')
print("Confusion Matrix:\n", confusion_matrix(y2,p))
print("Report:\n", classification_report(y2,p))