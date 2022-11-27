import numpy as np
import pylab


def ransac(data, model, n, k, t, d, return_all=False):
    """
    :param data: 数据集
    :param model: 假设模型
    :param n: 生成模型所需的最少样本点
    :param k: 最大迭代次数
    :param t: 阈值:作为判断点满足模型的条件
    :param d: 拟合较好时,需要的样本点最少的个数,当做阈值看待
    :param return_all:
    :return: 最优拟合解
    """
    iterations = 0
    best_fit = None
    best_error = np.inf  # 设置默认值
    best_inlier_indexes = None
    while iterations < k:
        maybe_indexes, test_indexes = random_partition(n, data.shape[0])
        print('test_indexes = ', test_indexes)
        maybe_inliers = data[maybe_indexes, :]  # 获取size(maybe_indexes)行数据(Xi,Yi)
        test_points = data[test_indexes]  # 若干行(Xi,Yi)数据点
        maybe_model = model.fit(maybe_inliers)  # 拟合模型
        test_err = model.get_error(test_points, maybe_model)  # 计算误差:平方和最小
        print('test_err = ', test_err < t)
        also_indexes = test_indexes[test_err < t]
        print('also_indexes = ', also_indexes)
        also_inliers = data[also_indexes, :]
        print('d = ', d)
        if len(also_inliers) > d:
            better_data = np.concatenate((maybe_inliers, also_inliers))  # 样本连接
            better_model = model.fit(better_data)
            better_errs = model.get_error(better_data, better_model)
            mean_error = np.mean(better_errs)  # 平均误差作为新的误差
            if mean_error < best_error:
                best_fit = better_model
                best_error = mean_error
                best_inlier_indexes = np.concatenate((maybe_indexes, also_indexes))  # 更新局内点,将新点加入
        iterations += 1
    if best_fit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return best_fit, {'inliers': best_inlier_indexes}
    else:
        return best_fit


def random_partition(n, n_data):
    """
    获取随机索引
    :param n: 分割数据数量
    :param n_data: 数据
    """
    all_indexes = np.arange(n_data)  # 获取n_data下标索引
    np.random.shuffle(all_indexes)  # 打乱下标索引
    indexes1 = all_indexes[:n]
    indexes2 = all_indexes[n:]
    return indexes1, indexes2


class LinearLeastSquareModel:
    """
    自定义线性模型
    """

    # 最小二乘求线性解,用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns):
        """
        :param input_columns: 输入列
        :param output_columns: 输出列
        """
        self.input_columns = input_columns
        self.output_columns = output_columns

    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        a = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        b = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        x, resides, rank, s = np.linalg.lstsq(a, b)  # residues:残差和
        return x

    def get_error(self, data, model):
        a = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        b = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        b_fit = np.dot(a, model)  # 计算的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum((b - b_fit) ** 2, axis=1)  # sum squared error per row
        return err_per_point


def main():
    # 生成理想数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    a_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20之间的500个数据:行向量
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率
    b_exact = np.dot(a_exact, perfect_fit)  # y = x * k

    # 加入高斯噪声,最小二乘能很好的处理
    a_noisy = a_exact + np.random.normal(size=a_exact.shape)  # 500 * 1行向量,代表Xi
    b_noisy = b_exact + np.random.normal(size=b_exact.shape)  # 500 * 1行向量,代表Yi

    # 添加"局外点"
    n_outliers = 100
    all_indexes = np.arange(a_noisy.shape[0])  # 获取索引0-499
    np.random.shuffle(all_indexes)  # 将all_indexes打乱
    outlier_indexes = all_indexes[:n_outliers]  # 100个0-500的随机局外点
    a_noisy[outlier_indexes] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi
    b_noisy[outlier_indexes] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi
    # setup model
    all_data = np.hstack((a_noisy, b_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列
    input_columns = range(n_inputs)  # 数组的第一列x:0
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1

    model = LinearLeastSquareModel(input_columns, output_columns)  # 类的实例化:用最小二乘生成已知模型

    linear_fit, resides, rank, s = np.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # run RANSAC 算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, return_all=True)

    sort_indexes = np.argsort(a_exact[:, 0])
    a_col0_sorted = a_exact[sort_indexes]  # 秩为2的数组

    pylab.plot(a_noisy[:, 0], b_noisy[:, 0], 'k.', label='data')  # 散点图
    pylab.plot(a_noisy[ransac_data['inliers'], 0], b_noisy[ransac_data['inliers'], 0], 'bx', label="RANSAC data")

    pylab.plot(a_col0_sorted[:, 0], np.dot(a_col0_sorted, ransac_fit)[:, 0], label='RANSAC fit')
    pylab.plot(a_col0_sorted[:, 0], np.dot(a_col0_sorted, perfect_fit)[:, 0], label='exact system')
    pylab.plot(a_col0_sorted[:, 0], np.dot(a_col0_sorted, linear_fit)[:, 0], label='linear fit')
    pylab.legend()
    pylab.show()


if __name__ == "__main__":
    main()
