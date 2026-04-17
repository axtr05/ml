import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.DataFrame({
    'Age':[45,54,60,48,52,46],
    'Sex':[1,0,1,0,1,0],
    'ChestPain':[3,2,1,2,3,1],
    'BP':[130,140,120,135,150,128],
    'Cholesterol':[230,250,240,220,260,210],
    'MaxHR':[150,140,130,160,120,170],
    'Target':[1,1,0,1,0,1]
})

x = df.drop('Target', axis=1)
y = df['Target']

xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=42)

lr = LogisticRegression()
dt = DecisionTreeClassifier()

lr.fit(xtr, ytr)
dt.fit(xtr, ytr)

print("LR:", accuracy_score(yte, lr.predict(xte)))
print("DT:", accuracy_score(yte, dt.predict(xte)))