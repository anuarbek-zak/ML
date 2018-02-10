import numpy as np
from numpy import corrcoef

data = np.loadtxt("data.txt", dtype=str, delimiter=',')
# splitting parameters and labels
parameters = data[:2, 1:-1]
labels = data[:2, -1]
print ('parameters:',parameters)
params_2 = []
for i in range(0,len(parameters)):
    params_2.append([])
    params_2[i].append(float((parameters[i][0])))
    params_2[i].append(float((parameters[i][1])))
    params_2[i].append(float((parameters[i][6])))
    params_2[i].append(float((parameters[i][12])))


cor = corrcoef(params_2[0],params_2[1])[0][1]
print ('params_2:',params_2)
print (cor)
