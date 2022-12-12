"""
变量更新
update=tf.assign(ref, value):当这个赋值被完成，该旧值ref才会被修改成value，即执行run()的操作。
"""

import tensorflow as tf


def func():
    ref_a = tf.Variable(tf.constant(1))
    ref_b = tf.Variable(tf.constant(2))
    update = tf.assign(ref_a, 10)
    ref_sum = tf.add(ref_a, ref_b)
    tf.global_variables_initializer()
    print('-' * 40)
    print('func:')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(ref_sum))
    print('-' * 40)


def func2():
    ref_a = tf.Variable(tf.constant(1))
    ref_b = tf.Variable(tf.constant(2))
    update = tf.assign(ref_a, 10)
    ref_sum = tf.add(ref_a, ref_b)
    tf.global_variables_initializer()
    print('-' * 40)
    print('func2:')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(update)
        print(sess.run(ref_sum))
    print('-' * 40)


def func3():
    state = tf.Variable(0, name='state')
    one = tf.constant(1)
    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('-' * 40)
        print('func3:')
        print(f"state:{sess.run(state)}")
        for i in range(5):
            sess.run(update)
            print(f"state:{sess.run(state)}")
        print('-' * 40)


if __name__ == '__main__':
    func()
    func2()
    func3()
