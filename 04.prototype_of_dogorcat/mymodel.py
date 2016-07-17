# -*- coding: utf-8 -*-

# 必要な物をインポート
import tensorflow as tf

class mymodel(object):

    #################################################
    # コンストラクタ
    #################################################
    def __init__(self, width, height, channels, classes):
        self.image_width = width
        self.image_height = height
        self.color_channels = channels
        self.classes = classes

    #################################################
    # 初期化された重みVariableを返す
    #################################################
    def weight_variable(self, shape):
        # 正規分布に従うランダム値を返す。stddevは標準偏差
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    #################################################
    # 初期化されたバイアスVariableを返す
    #################################################
    def bias_variable(self, shape):
        # 0.1固定
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    #################################################
    # 畳み込みを行い特徴マップを返す
    #################################################
    def conv2d(self, x, W):
        # stridesは次にフィルタリングする位置への移動量
        # paddingがSAMEの場合、入力(x)サイズと出力サイズは同じ
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    #################################################
    # 最大値プーリングの結果を返す
    #################################################
    def max_pool(self, x):
        # ksizeはpoolingするサイズ
        # stridesは次にpoolingする位置への移動量
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    #################################################
    # モデルを構築する
    #################################################
    def construct(self):

        # 入力データ格納用Variable作成
        num_elm_x = self.image_width * self.image_height * self.color_channels
        self.x = tf.placeholder(tf.float32, [None, num_elm_x])

        # 正解データ格納用Variable作成
        self.y_ = tf.placeholder(tf.float32, [None, self.classes])

        # First Convolutional Layer
        # フィルタのサイズ=5*5[pixels]を32個
        W_conv1 = self.weight_variable([5, 5, 3, 32])
        b_conv1 = self.bias_variable([32])

        # First Convolutional Layerの入力
        # 1d dimensionは画像数(-1は元サイズと同じの意味)
        # 2,3d dimensionは画像のwidth,heightに対応
        # 4d dimensionはcolor channlesに対応
        x_image = tf.reshape(self.x, [-1, self.image_width, self.image_height, self.color_channels])

        # First Convolutional Layerの計算
        # ReLU:入力が0以下なら0、0以上なら入力がそのまま出力になる
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool(h_conv1)

        # Second Convolutional Layer
        # フィルタのサイズ=5*5[pixels]
        # フィルタの個数は32個から64個へ増やす
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])

        # Seconde Convolutional Layerの計算
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool(h_conv2)

        # Densely Connected Layer(全結合層)
        # 入力を3次元から1次元のテンソルに変更
        # 出力数は1024
        W_fc1 = self.weight_variable([8 * 8 * 3 * 64, 1024])
        b_fc1 = self.bias_variable([1024])

        # Second Convolutional Layerの出力を
        # 入力を3次元から1次元のテンソルに変更
        h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 3 * 64])

        # h_pool2_flatとW_fc1を乗算し、バイアスを加算した値に
        # ReLUを適用する。
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # h_fc1に対してドロップアウトを適用する。
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Readout Leyer(出力層)
        W_fc2 = self.weight_variable([1024, self.classes])
        b_fc2 = self.bias_variable([self.classes])
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        # 誤差関数=クロスエントロピー
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(y_conv), reduction_indices=[1]))

        # Adam Algorithmで学習係数=0.0001
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.sess = tf.Session()
        init = tf.initialize_all_variables()
        self.sess.run(init)

    #################################################
    # 学習を行う
    #################################################
    def iteration(self, images, labels):

        # セッション生成
        # 1回の計算でMNISTデータ50個を使う。
        print(type(images))
        num_data = len(images)
        i = 0
        batch_size = 50
        while i < num_data:

            input_x = images[i:i+batch_size]
            input_y = labels[i:i+batch_size]
            print('size input_x[0] : %d' % len(input_x[0]))
            #if i % 10 == 0:
            #    train_accuracy = self.accuracy.eval(feed_dict={self.x: input_x, self.y_: input_y, self.keep_prob: 1.0})
            #    print("step %d, training accuracy %g" % (i, train_accuracy))

            # Dropout率=0.5で計算
            self.sess.run(self.train_step, feed_dict={self.x: input_x, self.y_: input_y, self.keep_prob: 0.5})

            i += batch_size
            if (i+batch_size) > num_data:
                batch_size = num_data - i
