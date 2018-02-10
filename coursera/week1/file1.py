import numpy as np

x = np.random.normal(loc=1,scale=10,size=(1000,500))
m = np.mean(x,axis=0)
s = np.std(x,axis=0)
x_norm = ((x-m)/s)
# print (x)

Z = np.array([[4, 5, 0],
             [1, 9, 3],
             [5, 1, 1],
             [3, 3, 3],
             [9, 9, 9],
             [4, 7, 1]])
r = np.sum(Z,axis=1)
print  (np.nonzero(r>10))

m1 = np.eye(3)
m2 = np.eye(3)
res = np.vstack((m1,m2))
print (res)