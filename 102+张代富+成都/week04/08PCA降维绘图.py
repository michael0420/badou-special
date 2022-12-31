import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

x, y = load_iris(return_X_y=True)
pca = PCA(n_components=2)
pca_x = pca.fit_transform(x)
fashion = [['r', 'b', 'g'], ['x', 'd', '.']]
for i in range(3):
    plt.scatter(pca_x[y == i][:, 0], pca_x[y == i][:, 1], c=fashion[0][i], marker=fashion[1][i])
plt.legend(labels=range(3))
plt.show()
