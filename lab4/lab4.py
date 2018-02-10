import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

data = load_iris()

model = GaussianNB()
X_train,X_test,Y_train,Y_test = train_test_split(data.data,data.target,test_size=0.30,random_state=45)
model.fit(X_train,Y_train)

prediction = model.predict(X_test)

print("accuracy:", accuracy_score(Y_test,prediction))
print("precision:", precision_score(Y_test,prediction, average=None))
print("recall:", recall_score(Y_test,prediction, average=None))
print("f1:", f1_score(Y_test,prediction, average=None))
print("classification_report:", classification_report(Y_test,prediction))
