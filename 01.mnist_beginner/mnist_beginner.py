# -*- coding: utf-8 -*-

# 必要なものインポート
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNISTのデータダウンロードor解凍
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 入力Variable(?*784)作成
x = tf.placeholder(tf.float32, [None, 784])

# 重みとバイアスVarialbe作成(0クリア)
# 784 = 28 * 28 [pixels]
# 10 = 手書き文字の種類(0〜9)
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 出力Variable
# tf.matmul : 行列と行列の乗算
# tf.nn.softmax : 値を0〜1にする関数
# ソフトマックス関数：尤度関数の一種
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 正解(MNISTデータ)Variable
y_ = tf.placeholder(tf.float32, [None, 10])

# クロスエントロピー：誤差関数の一種
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 最急降下法で、学習係数=0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    # 確率的勾配降下法
    # MNISTデータから100データ取得して計算させる。
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
    #print(sess.run(y_, feed_dict={x: batch_xs}))

# Evaluating Our Model
# test用データをインプットに実行した正解率を表示する。
# argmax()は最大値のインデックスを返す
# y_のインデックスと正解yのインデックスが合っているかカウントする。
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

