import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./data/train_data.csv')
X = data['X'].values
Y = data['Y'].values

n = data.shape[0]

# w b
w = (n * (X * Y).sum() - X.sum() * Y.sum()) / (n * (X ** 2).sum() - X.sum() ** 2)
b = Y.sum() / n - w * X.sum() / n
print(f"线性方程：y = {w}x + {b}")

# 画图
plt.scatter(X, Y, c='red', marker='x', label='actual')
plt.plot(X, w * X + b, c='orange', label='predict')
plt.legend()
plt.show()
