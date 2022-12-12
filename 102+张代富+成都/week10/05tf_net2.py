import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def get_data():
    _x = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
    noise = np.random.normal(0, 0.02, _x.shape)
    _y = np.square(_x) + noise
    return _x, _y


def net(data, out_shape):
    w = tf.Variable(tf.random_normal([data.shape.as_list()[1], out_shape]))
    b = tf.Variable(tf.zeros([1, data.shape[1]]))
    o = tf.matmul(data, w) + b
    return tf.nn.tanh(o)


x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

data_x, data_y = get_data()
result = net(net(x, 10), 1)

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
