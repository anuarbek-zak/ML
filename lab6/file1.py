import numpy as np
from sklearn.linear_model import LinearRegression
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

path = os.path.abspath('titanic.csv')
data = pd.read_csv(path)
data = data.dropna()
data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
data['Sex'] = data['Sex'].map({'female':1,'male':0})
x = data.drop('Survived',axis=1)
y = data['Survived']
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.30,random_state=1)
model =LinearRegression()

model.fit(X_train,Y_train)
y_predict = model.predict(X_test)
print ('y_pred',y_predict[:])
print ('y_test',Y_test[:])

# print (accuracy_score(Y_test,y_predict))
print (classification_report(Y_test,y_predict))
