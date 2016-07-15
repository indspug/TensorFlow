# -*- coding: utf-8 -*-

# 必要なものインポート
#import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 定数
MODEL_PATH="./model.ckpt"

# MNISTのデータダウンロードor解凍
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# チェックポイントの有無取得
ckpt = tf.train.get_checkpoint_state('./')

# 学習済モデル無し
#if not (os.path.exists(MODEL_PATH)) :
if not (ckpt) :
  
  # 入力Variable(?*784)作成
  x = tf.placeholder(tf.float32, [None, 784])

  # 重みとバイアスVarialbe作成(0クリア)
  # 784 = 28 * 28 [pixels]
  # 10 = 手書き文字の種類(0〜9)
  W = tf.Variable(tf.zeros([784, 10]), name="W")
  b = tf.Variable(tf.zeros([10]), name="b")

  # 出力Variable
  # tf.matmul : 行列と行列の乗算
  # tf.nn.softmax : 値を0〜1にする関数
  # ソフトマックス関数：尤度関数の一種
  y = tf.nn.softmax(tf.matmul(x, W) + b, "y")

  # 正解(MNISTデータ)Variable
  y_ = tf.placeholder(tf.float32, [None, 10])

  # クロスエントロピー：誤差関数の一種
  cross_entropy = -tf.reduce_sum(y_*tf.log(y))

  # 最急降下法で、学習係数=0.01
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
  init = tf.initialize_all_variables()

  # Saver生成
  saver = tf.train.Saver([W, b])
  
  sess = tf.Session()
  sess.run(init)
  
  for i in range(1000):
    # 確率的勾配降下法
    # MNISTデータから100データ取得して計算させる。
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
    #print(sess.run(y_, feed_dict={x: batch_xs}))

  # 学習したモデルをファイルに保存
  save_path = saver.save(sess, MODEL_PATH)
  print("Model saved in file: %s" % save_path)

# 学習済モデル有り
else:
  # 重みとバイアスVarialbe作成
  W = tf.Variable(tf.zeros([784, 10]), name="W")
  b = tf.Variable(tf.zeros([10]), name="b")
  #W = tf.Variable(..., name="W")
  #b = tf.Variable(..., name="b")

  # Session Saver生成
  sess = tf.Session()
  saver = tf.train.Saver()
 
  # 学習したモデルをファイルからロード
  saver.restore(sess, MODEL_PATH)
  
  # 入力Variable(?*784)作成
  x = tf.placeholder(tf.float32, [None, 784])

  # 出力Variable
  y = tf.nn.softmax(tf.matmul(x, W) + b, "y")

  # 正解(MNISTデータ)Variable
  y_ = tf.placeholder(tf.float32, [None, 10])

#

# Evaluating Our Model
# test用データをインプットに実行した正解率を表示する。
# argmax()は最大値のインデックスを返す
# y_のインデックスと正解yのインデックスが合っているかカウントする。
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("Accuracy %f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



