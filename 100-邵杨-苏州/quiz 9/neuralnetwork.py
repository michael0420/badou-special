import numpy as np
import scipy.special

class neuralnetwork:
    # 建立初始参数，输入节点、隐藏层节点数、输出节点数、学习率
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.i_nodes = inputnodes
        self.h_nodes = hiddennodes
        self.o_nodes = outputnodes
        self.l_r = learningrate
    # 建立链接之间的权重参数w矩阵，遵循正太分布，  均值0，  标准差为输出节点个数的-0.5次方，   输出节点个数的行数， 输入节点个数的列数  的矩阵
        np.random.seed(100)
        self.w_i_h = (np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes)))
        self.w_h_o = (np.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes)))

        self.activation_function = lambda x: scipy.special.expit(x)


    def train(self, inputs, targets):
        inputs = np.array(inputs, ndmin=2).T      #(输入节点个数,1)
        targets = np.array(targets, ndmin=2).T    #（输出节点个数，1）
        hidden_inputs = np.dot(self.w_i_h, inputs)   #（隐藏节点个数，1）
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.w_h_o, hidden_outputs)  #（输出节点个数，1）
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs   #（输出节点个数，1）
        # hideden_errors 代表误差的梯度值，区别于上面output_error
        hidden_errors = np.dot(self.w_h_o.T, output_errors*final_outputs*(1-final_outputs))  #（隐藏节点个数，1）

        self.w_h_o += self.l_r * np.dot(output_errors * final_outputs * (1-final_outputs), hidden_outputs.T)  #（输出个数，隐藏个数）
        self.w_i_h += self.l_r * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), inputs.T)   #（隐藏个数，输入个数）

        pass

    def query(self, inputs):
        hidden_inputs = np.dot(self.w_i_h, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.w_h_o, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs

input_nodes = 784
hidden_nodes = 225
output_nodes = 10
learning_rate = 0.05
n = neuralnetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("dataset/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
epochs = 10
for e in range(epochs):
    #把数据依靠','区分，并分别读入
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        #设置图片与数值的对应关系
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:", correct_number)
    #预处理数字图片
    inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    #让网络判断图片对应的数字
    outputs = n.query(inputs)
    #找到数值最大的神经元对应的编号
    label = np.argmax(outputs)
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

#计算图片判断的成功率
scores_array = np.asarray(scores)
print("rate = ", scores_array.sum() / scores_array.size)