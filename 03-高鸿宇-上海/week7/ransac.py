import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl

def ransac(data, model, n, k, t, d):
    
    def _random_split(n, n_samples):
        all_index = np.arange(n_samples)
        np.random.shuffle(all_index)
        return all_index[:n], all_index[n:]
        
    max_iter = 0
    best_err = np.inf
    best_fit = 0
    best_inlier_index = None
    
    while max_iter < k:
        maybe_inlier_index, test_index = _random_split(n, data.shape[0])
        maybe_inlier_data, test_data = data[maybe_inlier_index], data[test_index]
        maybe_model = model.fit(maybe_inlier_data)
        test_err = model.get_error(test_data, maybe_model)
        also_index = test_index[test_err < t]
        also_inlier_data = data[also_index]
        
        if (len(also_inlier_data) > d):
            inlier_index = np.concatenate((maybe_inlier_index, also_index))
            inlier_data = data[inlier_index]
            inlier_model = model.fit(inlier_data)
            current_error = np.mean(model.get_error(inlier_data, inlier_model))
            if (current_error < best_err):
                best_err = current_error
                best_fit = inlier_model
                best_inlier_index = inlier_index
        max_iter += 1
    
    return best_fit, best_inlier_index

class MyLinear:
    def __init__(self, input_columns, output_columns) -> None:
        self.input_columns = input_columns
        self.output_columns = output_columns
        
    def fit(self, data):
        X = data[:, self.input_columns]
        Y = data[:, self.output_columns]
        k, _, _, _ = sl.lstsq(X, Y)
        return k
    
    def get_error(self, x, model):
        X = x[:, self.input_columns]
        Y = x[:, self.output_columns]
        Y_hat = X @ model
        err = np.sum((Y - Y_hat) ** 2, axis=1)
        return err

if __name__ == "__main__":
    n_sample = 500
    n_inputs, n_outputs = 1, 1
    X = 20 * np.random.random((n_sample, n_inputs))
    K = 60 * np.random.normal(size=(n_inputs, n_outputs))
    Y = X @ K
    
    #对样本点添加噪声
    X_noise = X + np.random.normal(size=X.shape)
    Y_noise = Y + np.random.normal(size=Y.shape)
    
    # 将其中的若干个点设置为局外点
    n_outliers = 100
    all_index = np.arange(n_sample)
    np.random.shuffle(all_index)
    outlier_index = all_index[:n_outliers]
    X_noise[outlier_index] = 20 * np.random.random((n_outliers, n_inputs))
    Y_noise[outlier_index] = 50 * np.random.normal(size=(n_outliers, n_outputs))
    
    input_columns, output_columns = [0], [1]
    
    # 将X, Y拼接起来做成data（500,2）
    data = np.hstack((X_noise, Y_noise))
    
    model = MyLinear(input_columns, output_columns)
    
    # lstsq方法输入必须为二维向量
    linear_fit, _, _, _ = sl.lstsq(data[:,input_columns], data[:,output_columns])
    
    n, k, t, d = 50, 1000, 7e3, 300
    ransac_fit, ransac_index = ransac(data, model, n, k, t, d)
    
    sort_index = np.argsort(X[:, 0])
    sorted_X = X[sort_index]
    
    plt.figure()
    plt.scatter(X_noise[:, 0], Y_noise[:, 0], color='black', label='original data')
    plt.scatter(X_noise[outlier_index, 0], Y_noise[outlier_index, 0], color='red', label='outlier data')
    plt.scatter(X_noise[ransac_index, 0], Y_noise[ransac_index, 0], color='blue', marker='x', label='ransac data')
    plt.plot(sorted_X[:, 0], (sorted_X@K)[:, 0], label='exact system')
    plt.plot(sorted_X[:, 0], (sorted_X@linear_fit)[:, 0], label='linear fit')
    plt.plot(sorted_X[:, 0], (sorted_X@ransac_fit)[:, 0], label='RANSAC fit')
    plt.legend()
    plt.show()