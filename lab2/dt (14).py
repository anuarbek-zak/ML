import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *

data = np.loadtxt("data.txt", dtype=int, delimiter=',')
# splitting parameters and labels
parameters = data[:, 1:10]
labels = data[:, -1]

# calculating sizes of train and test data
rows = len(parameters)
train_num = int(round(rows * .7, 0));
test_num = rows - train_num
# splitting train and test data
train_params = parameters[:train_num]
train_labels = labels[:train_num]
test_params = parameters[train_num:]
test_labels = labels[train_num:]

# building a model
tree = DecisionTreeClassifier()
tree.fit(train_params, train_labels)

prediction = tree.predict(test_params)

print("accuracy:", accuracy_score(test_labels, prediction))
print("precision:", precision_score(test_labels, prediction, average=None))
print("recall:", recall_score(test_labels, prediction, average=None))


print(np.sum(prediction == test_labels), len(test_labels))