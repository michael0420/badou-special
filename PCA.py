import numpy as np
def PCA(data,a):
    m,n=data.shape
    for i in range(n):
        data[:,i]=data[:,i]-sum(data[:,i])/m
    xiefangchajuzhen=np.dot(data.T,data)/m
    tezhengzhi,tezhengxiangliang=np.linalg.eig(xiefangchajuzhen)
    ind=np.argsort(-1*tezhengzhi)
    bianhuanjuzheng=tezhengxiangliang[:,ind[0]]
    if a>1:
        for i in range(1,a):
            bianhuanjuzheng=[bianhuanjuzheng,tezhengxiangliang[:,ind[i]]]
    bianhuanjuzheng=np.transpose(bianhuanjuzheng)
    jiangweijuzhen=np.dot(data,bianhuanjuzheng)
    return jiangweijuzhen
data=np.array([[1,2,3],
               [2,3,4],
               [3,4,5],
               [5,6,7]],dtype=float)
jiangwei=PCA(data,2)
print(jiangwei)
