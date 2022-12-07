import tensorflow as tf

a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)  # 2 * 3
w = tf.constant([[0.5, 1]], dtype=tf.float32)  # 1 * 2
with tf.Session() as sess:
    print(tf.matmul(w, a).eval())  # 1 * 3
