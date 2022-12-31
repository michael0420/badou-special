import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


class RandomSampleConsensus:
    def __init__(self, model, iter_n, thr):
        self.sample_n = 500  # 500样本数据
        self.sample_range = [0, 50]  # 样本范围
        self.sample_data = None
        self.sample_model = None
        self.exact_data = None
        self.iter_n = iter_n  # 迭代次数
        self.thr = thr  # 误差阈值
        self.linear_params = None
        self.best_params = None
        self.best_err = np.Inf
        self.better_num = 300  # 内点数量阈值
        self.interior_point = None
        self.model = model

    def generate_sample_data(self):
        """
        生成 500个 样本数据
        y = kx + b
        :return:
        """
        r = self.sample_range
        x_data = r[0] + (r[1] - r[0]) * np.random.random((self.sample_n, 1))

        k = 50 * np.random.random((1, 1))
        b = 50 * np.random.random((1, 1))
        y_data = np.dot(x_data, k) + b
        self.exact_data = {"x": x_data, "y": y_data}
        # self.sample_model = {"k": k, "b": b}
        # 加入噪声：
        x_data = x_data + np.random.normal(size=x_data.shape)
        y_data = y_data + np.random.normal(size=y_data.shape)
        self.sample_data = {"x": x_data, "y": y_data}

        # 加入局外点
        # 百分之20的数据作为局外点
        outlier_point = int(self.sample_n * 0.2)
        index = [i for i in range(self.sample_n)]
        np.random.shuffle(index)
        index = index[0:outlier_point]
        x_data[index] = x_data[index] + 50 * np.random.normal(size=(outlier_point, 1))
        y_data[index] = y_data[index] + 50 * np.random.normal(size=(outlier_point, 1))

    def get_random_data(self):
        n = self.model.min_data_n
        index = [i for i in range(self.sample_n)]
        np.random.shuffle(index)
        index = index[0:n]
        x_data = self.sample_data['x']
        y_data = self.sample_data['y']
        return x_data[index], y_data[index]

    def random_sample_consensus(self):
        iter_c = 0
        while iter_c < self.iter_n:
            iter_c += 1
            x, y = self.get_random_data()
            self.model.fit(x, y)
            # err = self.model.get_err(x,y)
            err = self.model.get_err(self.sample_data['x'], self.sample_data['y'])
            index = err < self.thr
            x_data = self.sample_data["x"][index]
            y_data = self.sample_data["y"][index]
            if len(x_data) > self.better_num:
                self.best_params = self.model.params
                self.best_err = err
                self.best_data = {"x": x_data, "y": y_data}

    def linear_fit(self):
        A = np.hstack([self.sample_data["x"] ** 0, self.sample_data["x"] ** 1])
        x, residues, rank, s = linalg.lstsq(A, self.sample_data["y"])  # residues:残差和
        self.linear_params = x
        pass

    def display_data(self):
        plt.plot(self.sample_data["x"], self.sample_data["y"], 'k.', label='data')

        plt.plot(self.best_data["x"], self.best_data["y"], 'bx', label='ransac data')
        sort_idx = np.argsort(self.exact_data["x"], axis=0)
        sort_idx = sort_idx.T[0]
        plt.plot(self.exact_data["x"][sort_idx], self.exact_data["y"][sort_idx], label='exact lines')

        plt.plot(self.exact_data["x"][sort_idx],
                 np.dot(np.hstack([self.exact_data["x"][sort_idx] ** 0, self.exact_data["x"][sort_idx]]),
                        self.best_params), label='ransac lines')

        plt.plot(self.exact_data["x"][sort_idx],
                 np.dot(np.hstack([self.exact_data["x"][sort_idx] ** 0, self.exact_data["x"][sort_idx]]),
                        self.linear_params), label='lines fit')
        plt.legend()
        plt.show()

    def run(self):
        self.generate_sample_data()
        self.random_sample_consensus()
        self.linear_fit()
        self.display_data()
        pass


class LineModel:
    min_data_n = 2

    def __init__(self):
        self.params = None
        pass

    def fit(self, X, Y):
        """模型为 y=kx+b"""
        A = np.hstack([X ** 0, X ** 1])
        x, residues, rank, s = linalg.lstsq(A, Y)  # residues:残差和
        self.params = x

    def get_err(self, X, Y):
        # A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        # B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        A = np.hstack([X ** 0, X ** 1])
        X_fit = np.dot(A, self.params)  # 计算的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum((Y - X_fit) ** 2, axis=1)  # sum squared error per row
        return err_per_point


if __name__ == '__main__':
    random_sample_consensus = RandomSampleConsensus(LineModel(), iter_n=1000, thr=5000)
    random_sample_consensus.run()
