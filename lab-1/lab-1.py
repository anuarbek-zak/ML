import numpy	 as np
from pandas import *
import os

path = os.path.abspath("wine.csv")
data = read_csv(path, delimiter=",")

print (data.columns)

X = data.values[::, 1:14]
y = data.values[::, 0:1]

from sklearn.cross_validation import train_test_split as train
X_train, X_test, y_train, y_test = train(X, y, test_size=0.6)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
print(2+2)