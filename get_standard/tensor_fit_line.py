# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

# (y1) = (w11 w12) * (x1) + (b1)
# (y2)   (w21 w22)   (x2)   (b2)
# (w1,w2),(b1,b2)を求める

# 正解データの作成
w_data = np.array([[0.1, 0.1], [0.1, 0.1]])
b_data = np.array([0.3, 0.3])
x_data = np.random.rand(100,2,1).astype(np.float32)
y_data = w_data * x_data + b_data

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but Tensorflow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([2,2], -1.0, 1.0))
b = tf.Variable(tf.zeros([2]))
y = W * x_data + b

# Minimize the mean squared errors.
# loss:y-ydataの二乗の平均値
loss = tf.reduce_mean(tf.square(y - y_data))

# Training用クラスの生成
# Gradient Descent:最急降下法, 重み=0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)
#print(sess.run(W), sess.run(b))
#print(sess.run(y), sess.run(loss))

# Fit the line.
for step in range(1001):
    sess.run(train)
    if step % 100 == 0:
        print(step, sess.run(W), sess.run(b))

# Learns best fit is W: [[0.1,0.1],[0.1,0.1]], b: [0.3,0.3]

