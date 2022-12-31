# ==================================3:实现PCA==================================
#整个PCA降维过程其实就是一个实对称矩阵对角化的过程

# PCA降维是通过变换坐标系，来尽可能的减少信息损失
# 矩阵X每一行就是一条记录，每一列就是特征
# 我们想要对它降维，也就是想让它从m×n维降到m×r维(n>r)

# 那到底删除哪几列损失信息才最少？或者通过乘以一个满秩矩阵映射到其他坐标系让它特征变少？
# 其实无论你是直接在原数据上删除某几列还是想通过坐标变换映射到其他基坐标系，都可以通
# 过让X右乘一个满秩矩阵的形式来实现

# 右乘一个满秩矩阵来进行坐标变换，其实就是数据向这些新的基向量上投影，而这个满秩矩阵
# P就是新的基向量组成的矩阵：


# PCA具体算法步骤
# 设有M个N维数据:

    # 将原始数据按列组成N行   M列矩阵X

    # 将X的每一行进行零均值化（中心化），即减去每一行的均值

    # 求出X的协方差矩阵C

    # 求出协方差矩阵C的特征值及对应的特征向量，C的特征值就是Y的每维元素的方差，也是D的对角线元素，从大到小沿对角线排列构成D。
    #协方差矩阵的特点
    #协方差矩阵计算的是不同维度之间的协方差，，而不是不同样本之间的
    #样本矩阵的每一行是一个样本，每一列是一个维度，所以我们要按列计算均值
    #协方差矩阵的对角线就是各个维度上的方差

    # 将特征向量按对应特征值大小从上到下按行排列成矩阵，根据实际业务场景，取前R行组成矩阵P

    # Y=PX即为降到R维后的目标矩阵




# 使用PCA求样本矩阵X的K阶降维矩阵Z
import numpy as np


class CPCA(object):
    #用PCA求样本矩阵X的K阶降维矩阵Z--（Z降维后的矩阵）（K为阶数）
    #ote:请保证输入的样本矩阵X shape=(m, n)，m行样例，n个特征


    def __init__(self, X, K):

        #:param X,训练样本矩阵X
        #:param K,X的降维矩阵的阶数，即X要特征降维成k阶

        self.X = X                        # 样本矩阵X
        self.K = K                        # K阶降维矩阵的K值
        self.centrX = []                  # 矩阵X的中心化（零均值化）
        self.C = []                       # 样本集的协方差矩阵C
        self.U = []                       # 样本矩阵X的降维转换矩阵
        self.Z = []                       # 样本矩阵X的降维矩阵Z

        self.centrX = self.central()
        self.C = self.cov()              #样本集的协方差矩阵C
        self.U = self._U()                #样本矩阵X的降维转换矩阵Z=X*U
        self.Z = self._Z()                # Z=XU求得

    def central(self):               #样本矩阵X的中心化
        print('样本矩阵X:\n', self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 样本集的特征均值（每一列）
        #X为10行3列，X.T为3行10列，attr用于遍历X.T的每一行，也就是X的一列，大小为(10，)

        print('样本集的特征均值:\n', mean)
        centrX = self.X - mean            #样本集的中心化
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX


    def cov(self):                       #样本矩阵X的协方差矩阵
        # 样本集的样例总数
        ns = np.shape(self.centrX)[0]
        #shape的0维度就是矩阵的行数，就是样本数。(行是样本数，列则是维度数)

        # 样本矩阵的协方差矩阵C
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)
        #centrX.T代表的是矩阵的列（维度）中心化，np.dot函数表示相乘

        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _U(self):                         #样本矩阵X的降维转换矩阵U,shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度
        # 先求X的协方差矩阵C的特征值和特征向量
        a, b = np.linalg.eig(self.C)
        # 计算方阵的特征值和特征向量，numpy提供了接口eig，直接调用就行

        # 特征值赋值给a，对应特征向量赋值给b。函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
        #np.linalg.eig函数求协方差矩阵的特征值和特征向量

        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)

        # 给出特征值降序的topK的索引序列
        ind = np.argsort(-1 * a)
        #np.argsort(a, axis=-1, kind='quicksort', order=None)
        #函数功能：将a中的元素从小到大排列，提取其在排列前对应的index(索引)输出。
        #（此时-1*a表示的是降序排列）

        # 构建K阶降维的降维转换矩阵U
        UT = [b[:, ind[i]] for i in range(self.K)]
        #UT = [b[:, ind[i]] for i in range(self.K)]其中ind是排好序的索引数组，
        #值为[0,1,2]，ind[i]取对应元素，b[:,ind[i]]代表取ind[i]对应的列
        #a是特征值，b是特征向量，一个特征值对应一个特征向量；ind是根据a特征值排序后的索引
        # 值数组，通过UT = [b[:, ind[i]] for i in range(self.K)]这句则是取出a排序后
        # 对应的b特征向量。K为阶数，代表取b所有特征向量中的前K个

        U = np.transpose(UT)
        #ranspose函数主要用来转换矩阵的维度。二维直接就是转置

        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def _Z(self):                         #样本矩阵按照Z=XU求降维矩阵Z，shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))        #（10,3）
        print('U shape:', np.shape(self.U))        #（3,2）
        print('Z shape:', np.shape(Z))             #（10,2）
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__ == '__main__':
    # 10样本3特征的样本集, 行为样例，列为特征维度
    X = np.array([[13, 45, 59],
                  [17, 45, 63],
                  [25, 635, 62],
                  [56, 4, 25],
                  [42, 56, 41],
                  [6, 56, 65],
                  [16, 78, 34],
                  [6, 65, 56],
                  [65, 16, 257],
                  [35, 78, 76]])
    K = np.shape(X)[1] - 1                  #实现降维（列数-1）
    #多维也是转置，shape的1维是列数，-1就是k取列数 - 1，就是降1维
    #np.shape(X)的结果是一个元组，值为(10,3)，np.shape(X)[1]代表取元组的第二个元素，
    #也就是3。np.shape(X)[1]-1则是3-1=2

    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = CPCA(X, K)


#
# import numpy as np
# from sklearn.decomposition import PCA
#
# X = np.array([[-3,3,66,-1], [-2,7,68,-1], [-5,9,55,-2], [3,22,16,1], [2,23,42,6], [5,5,73,1]])  #导入数据，维度为4
# pca = PCA(n_components=2)   #降到2维
# pca.fit(X)                  #用X拟合模型
# newX=pca.fit_transform(X)   #用X拟合模型得到降维后的数据
# PCA(copy=True, n_components=2, whiten=False)
# # n_components	    整数型，浮点型是或者’mle’。如果这个参数没有设定，那么所有原始维度都将保留。如果这里是’mle’且svd_solver == ‘full’，那函数将会自己推测维度。
# # copy	            布尔型，默认True。如果是False，那传递给fit函数的数据将被覆盖。
# # whiten         	布尔型，默认为False。是否消除矢量之间的相关性
#
# print(pca.explained_variance_ratio_)  #输出贡献率
# #它代表降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分。
#
# print(newX)                  #输出降维后的数据


# sklearn.decomposition.PCA算法解析与应用
#主成分分析（PCA）：就是在尽可能保留数据特征的情况下，降低数据的维度。
#如何实现：在sklearn.decomposition.PCA中，使用的是SVD（奇异值分解）的方法进行降维。

#SVD的基本思想：
# 设有一个m×n的矩阵A（设m<n），可以将它分解为Am×n=Um×m· Λ ·Vn×n。其中U和V是正交矩阵，Λ是对
# 角矩阵。这是所有矩阵都具有的性质。# 因此可以将A分解A=λ1UV+λ2UV+…+λmUV=A1+A2+…+Am
# 这其中有的Ai小，对A整体的影响小，可以忽略，将m个Ai变成k个Ai对整体的影响也不大。这样就实现了降维。

#算法参数
# sklearn.decomposition.PCA(n_components=None, *, copy=True, whiten=False, svd_solver=‘auto’,
# tol=0.0, iterated_power=‘auto’, random_state=None)

# 参数	                     功能
# n_components	    整数型，浮点型是或者’mle’。如果这个参数没有设定，那么所有原始维度都将保留。如果这里是’mle’且svd_solver == ‘full’，那函数将会自己推测维度。
# copy	            布尔型，默认True。如果是False，那传递给fit函数的数据将被覆盖。
# whiten         	布尔型，默认为False。是否消除矢量之间的相关性
# svd_solver        {‘auto’, ‘full’, ‘arpack’, ‘randomized’},，默认’auto’。运行不同的求解器，‘auto’是自动设定，‘full’是求解出全部特征值，'arpack’是求解出n_components个特征值。
# tol	            浮点型，默认0.0。设定特征值的公差范围。
# iterated_power	整数型，或者’auto’，默认‘auto’。迭代的次数。
# random_state	    整数型，RandomState实例或无，默认=无。当使用’arpack’或’randomized’求解器时使用。在多个函数调用之间传递int以获得可重现的结果。

# 方法
# 项目	                                Value
# fit(X[, y])	                用X拟合模型
# fit_transfrom(X[, y])	        使用X拟合模型
# get_covariance()	            用生成模型计算数据协方差
# get_params([])	            获取此估计值的参数
# get_precision()	            用生成模型计算数据精度矩阵
# inverse_transform(X)	        将数据转换回其原始空间
# score(X[, y])	                返回所有样本的平均对数似然率
# sore_samples(X)	            返回每个样本的对数似然率
# set_params(**)	            设置此估算器的参数
# transform(X)	                对X应用降维

#俩个PCA的类
# 第一个是explained_variance_，它代表降维后的各主成分的方差值。方差值越大，则说明越是
# 重要的主成分。

# 第二个是explained_variance_ratio_，它代表降维后的各主成分的方差值占总方差值的比例，
# 这个比例越大，则越是重要的主成分。

# n_components：PCA降维后的特征维度数目。也可以指定主成分的方差和所占的最小比例阈值，
# 让PCA类去根据样本特征方差来决定降维到的维度数。还可以将参数设置为"mle", 此时PCA类
# 会用MLE算法根据特征的方差分布情况去选择一定数量的主成分特征来降维。
# 默认n_components=min(样本数，特征数)。
