from numpy import corrcoef
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

def from_array():
    x = [1, 2, 3, 4, 5, 1, 2, 3]
    y = [1, 1, 2, 4, 3, 3, 4, 3]
    plt.scatter(x,y)
    plt.show()

def built_in_data():
    bc = load_breast_cancer()
    print(len(list(bc.feature_names)))
    mean_radius = bc.data[0, :]
    mean_texture = bc.data[1, :]
    print (len(mean_radius))
    print (mean_texture)
    plt.scatter(mean_radius,mean_texture)
    plt.show()

built_in_data()
# from_array()