import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 输入
data_x = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, data_x.shape)
data_y = np.square(data_x) + noise

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 中间层
layer1_w = tf.Variable(tf.random_normal([1, 10]))
layer1_b = tf.Variable(tf.zeros([1, 1]))
layer1_out = tf.matmul(x, layer1_w) + layer1_b
layer1_out_activate = tf.nn.tanh(layer1_out)

# 输出层
layer2_w = tf.Variable(tf.random_normal([10, 1]))
layer2_b = tf.Variable(tf.zeros([1, 1]))
layer2_out = tf.matmul(layer1_out_activate, layer2_w) + layer2_b
result = tf.nn.tanh(layer2_out)

# loss
loss = tf.reduce_mean(tf.square(y - result))
# 优化器
step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epochs = 2000
    # train
    for i in range(1, epochs + 1):
        sess.run(step, feed_dict={x: data_x, y: data_y})
    # prediction
    prediction = sess.run(result, feed_dict={x: data_x})

# 绘图
plt.scatter(data_x, data_y)
plt.plot(data_x, prediction, c='r', lw=10)
plt.show()