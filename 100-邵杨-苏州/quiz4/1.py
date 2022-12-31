import numpy as np
a = [2, -1, 4, 0]
b = sorted(a, reverse=True)
c = []
for i in b:
    c.append(a.index(i))

eigenvector = [[-0.30253213, -0.87499307, -0.37797014],
               [-0.86718533,  0.08811216,  0.49012839],
               [ 0.39555518, -0.47604975,  0.78543792]]

eigenvalue = [335.15738485,  95.32771231,  32.63712506]
index = []

eigenvalue1 = sorted(eigenvalue, reverse=True)  #特征值从大到小
for i in list(eigenvalue1):
    index.append(list(eigenvalue).index(i))  #特征值对应原位置索引
print(index)
eigenvectorT = [np.array(eigenvector)[:, index[i]] for i in range(2)]
eigenvector1 = np.transpose(eigenvectorT)


print(eigenvector1)