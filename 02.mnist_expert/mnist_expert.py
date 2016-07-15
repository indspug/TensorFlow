# -*- coding: utf-8 -*-

# 必要な物をインポート
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#################################################
# 初期化された重みVariableを返す
#################################################
def weight_variable(shape):
  # 正規分布に従うランダム値を返す。stddevは標準偏差
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#################################################
# 初期化されたバイアスVariableを返す
#################################################
def bias_variable(shape):
  # 0.1固定
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#################################################
# 畳み込みを行い特徴マップを返す
#################################################
def conv2d(x, W):
  # stridesは次にフィルタリングする位置への移動量
  # paddingがSAMEの場合、入力(x)サイズと出力サイズは同じ
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#################################################
# 2*2[pixels]の最大値を返す
#################################################
def max_pool_2x2(x):
  # ksizeはpoolingするサイズ
  # stridesは次にpoolingする位置への移動量
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#################################################
# main
#################################################
if __name__ == '__main__':
  # MNISTのデータダウンロードor解凍
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  
  # 入力(MNISTデータ)格納用Variable(?*784)作成
  x = tf.placeholder(tf.float32, [None, 784])
  
  # 正解(MNISTデータ)格納用Variable(?*10)作成
  y_ = tf.placeholder(tf.float32, [None, 10])

  # First Convolutional Layer
  # フィルタのサイズ=5*5[pixels]を32個
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  
  # First Convolutional Layerの入力
  # 1d dimensionは画像数(-1は元サイズと同じの意味)
  # 2,3d dimensionは画像のwidth,heightに対応
  # 4d dimensionはcolor channlesに対応
  x_image = tf.reshape(x, [-1,28,28,1])
  
  # First Convolutional Layerの計算
  # ReLU:入力が0以下なら0、0以上なら入力がそのまま出力になる
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)
  
  # Second Convolutional Layer
  # フィルタのサイズ=5*5[pixels]
  # フィルタの個数は32個から64個へ増やす
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  
  # Seconde Convolutional Layerの計算
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)
  
  # Densely Connected Layer(全結合層)
  # 入力を3次元から1次元のテンソルに変更
  # 出力数は1024
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])
  
  # Second Convolutional Layerの出力を
  # 入力を3次元から1次元のテンソルに変更
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

  # h_pool2_flatとW_fc1を乗算し、バイアスを加算した値に
  # ReLUを適用する。
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # h_fc1に対してドロップアウトを適用する。
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) 

  # Readout Leyer(出力層)
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  # 誤差関数=クロスエントロピー
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

  # Adam Algorithmで学習係数=0.0001
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Session生成
  sess = tf.InteractiveSession()
  sess.run(tf.initialize_all_variables())

  for i in range(2000):
    # 1回の計算でMNISTデータ50個を使う。
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
      print("step %d, training accuracy %g"%(i, train_accuracy))
    # Dropout率=0.5で計算
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  # テスト用データで実行し正解率を表示
  # サンプル通りだとGPUメモリが足らなくなるので変更
  #print("test accuracy %g"%accuracy.eval(feed_dict={
  #  x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
  tacc = 0
  tbatchsize = 1000
  train_size = len(mnist.test.images)
  for i in range(0, train_size, tbatchsize):
    acc = accuracy.eval(feed_dict={x: mnist.test.images[i:i+tbatchsize], 
                                   y_: mnist.test.labels[i:i+tbatchsize], 
                                       keep_prob: 1.0})
    #print "test accuracy %d = %g" % (i, acc)
    tacc += acc * tbatchsize

  tacc /= train_size
  print "test accuracy = %g" % tacc 


