import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv("falldetection.csv")

X = pd.get_dummies(data[['TIME', 'SL', 'EEG', 'CIRCLUATION']])
y = pd.DataFrame(data['ACTIVITY'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

model = GaussianNB()
model.fit(X_train, y_train)

prediction= model.predict(X_test)
print (prediction)

print(accuracy_score(y_test, prediction))

