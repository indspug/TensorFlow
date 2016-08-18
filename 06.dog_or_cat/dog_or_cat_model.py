# -*- coding: utf-8 -*-

# DogOrCatModel.py

# import
from abstract_model import AbstractModel
import tensorflow as tf
import random

# 定数
#FILTER_SIZE1 = 5        # 畳込層1のフィルタサイズ
#FILTER_SIZE2 = 5        # 畳込層2のフィルタサイズ
#FILTER_NUM1 = 32        # 畳込層1のフィルタ数
#FILTER_NUM2 = 64        # 畳込層2のフィルタ数
#WIDTH_RATIO = 4         # 入力画像と全結合層手前の画像高さのサイズ比
#HEIGHT_RATIO = 4        # 入力画像と全結合層手前の画像高さのサイズ比
RATE_DROPOUT = 0.5      # ドロップアウト率
STEP_PRINT_ACC = 5     # 正解率を表示する頻度

class DogOrCatModel(AbstractModel):

    #################################################
    # 初期化された重みVariableを返す
    #################################################
    def weight_variable(self, shape, name=None):
        # 正規分布に従うランダム値を返す。stddevは標準偏差
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial, name)

    #################################################
    # 初期化されたバイアスVariableを返す
    #################################################
    def bias_variable(self, shape, name=None):
        # 0.1固定
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name)

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
    # Variableを初期化する
    #################################################
    def initialize_variables(self):

        # 入力データ格納用Variable作成
        num_elm_x = 28 * 28 * self.color_channels
        self.x = tf.placeholder(tf.float32, [None, num_elm_x])

        # 正解データ格納用Variable作成
        self.y_ = tf.placeholder(tf.float32, [None, self.num_classes])

        # First Convolutional Layerの入力
        x_image = tf.reshape(self.x, [-1, 28, 28, self.color_channels])

        # First Convolutional Layer
        with tf.variable_scope('conv1') as scope:
            self.W_conv1 = self.weight_variable([5, 5, 3, 32], name='W_conv1')
            self.b_conv1 = self.bias_variable([32], name='b_conv1')
            h_conv1 = tf.nn.conv2d(x_image, self.W_conv1, strides=[1, 1, 1, 1], padding='SAME')
            h_relu1 = tf.nn.relu(h_conv1 + self.b_conv1)
        h_pool1 = tf.nn.max_pool(h_relu1, ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1], padding='SAME')
        #h_norm1 = tf.nn.lrn(h_pool1, 5, bias=1.0, alpha=0.0001, beta=0.75)

        # Second Convolutional Layer
        with tf.variable_scope('conv2') as scope:
            self.W_conv2 = self.weight_variable([5, 5, 32, 64], name='W_conv2')
            self.b_conv2 = self.bias_variable([64], name='b_conv2')
            h_conv2 =  tf.nn.conv2d(h_pool1, self.W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            h_relu2 = tf.nn.relu(h_conv2 + self.b_conv2)
        h_pool2 = tf.nn.max_pool(h_relu2, ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1], padding='SAME')
        #h_norm2 = tf.nn.lrn(h_pool2, 5, bias=1.0, alpha=0.0001, beta=0.75)

        # Third Convolutional Layer
        #with tf.variable_scope('conv3') as scope:
            #self.W_conv3 = self.weight_variable([3, 3,  128,  256], name='W_conv3')
            #self.b_conv3 = self.bias_variable([256], name='b_conv3')
            #h_conv3 =  tf.nn.conv2d(h_norm2, self.W_conv3, strides=[1, 1, 1, 1], padding='VALID')
            #h_relu3 = tf.nn.relu(h_conv3 + self.b_conv3)
        #h_pool3 = tf.nn.max_pool(h_relu3, ksize=[1, 2, 2, 1],
                                 #strides=[1, 2, 2, 1], padding='VALID')
        # Fourth Convolutional Layer
        #with tf.variable_scope('conv4') as scope:
            #self.W_conv4 = self.weight_variable([3, 3,  384,  384], name='W_conv4')
            #self.b_conv4 = self.bias_variable([384], name='b_conv4')
            #h_conv4 =  tf.nn.conv2d(h_relu3, self.W_conv4, strides=[1, 1, 1, 1], padding='SAME')
            #h_relu4 = tf.nn.relu(h_conv4 + self.b_conv4)

        # Fifth Convolutional Layer
        #with tf.variable_scope('conv5') as scope:
            #self.W_conv5 = self.weight_variable([3, 3,  384,  256], name='W_conv5')
            #self.b_conv5 = self.bias_variable([256], name='b_conv5')
            #h_conv5 =  tf.nn.conv2d(h_relu4, self.W_conv5, strides=[1, 1, 1, 1], padding='SAME')
            #h_relu5 = tf.nn.relu(h_conv5 + self.b_conv5)
        #h_pool5 = tf.nn.max_pool(h_relu5, ksize=[1, 3, 3, 1],
        #                        strides=[1, 2, 2, 1], padding='VALID')
        h_pool5_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

        # Densely Connected Layer(全結合層)
        with tf.variable_scope('fc1') as scope:
            self.W_fc1 = self.weight_variable([7 * 7 * 64, 1024], name='W_fc1')
            self.b_fc1 = self.bias_variable([1024], name='b_fc1')
            h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat , self.W_fc1) + self.b_fc1)

        # ドロップアウトを使用する場合のドロップアウト率。
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Readout Leyer(出力層)
        with tf.variable_scope('fc2') as scope:
            self.W_fc2 = self.weight_variable([1024, self.num_classes], name='W_fc2')
            self.b_fc2 = self.bias_variable([self.num_classes], name='b_fc2')
            self.y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, self.W_fc2) + self.b_fc2)

    #################################################
    # ファイルに読み書きするVariableを取得する。
    #################################################
    def get_saveable_variables(self):
        return([self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2,
                #self.W_conv3, self.b_conv3, self.W_conv4, self.b_conv4,
                #self.W_conv5, self.b_conv5,
                self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]
        )

    #################################################
    # クロスエントロピーを返す
    # [Returns]:
    #    cross_entropy:クロスエントロピー(誤差関数)
    #################################################
    def get_cross_entropy(self):
        # 誤差関数=クロスエントロピー
        cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(self.y_ * tf.log(self.y_conv), reduction_indices=[1], name='sum'), name='cross_entropy')
        return(cross_entropy)

    #################################################
    # Train Stepを返す
    # [Args]:
    #    cross_entropy:
    # [Returns]:
    #    train_step:
    #################################################
    def get_train_step(self, cross_entropy):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, name='train_step')
        #train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy, name='train_step')
        #train_step = tf.train.MomentumOptimizer(1e-2, 1e-2).minimize(cross_entropy, name='train_step')
        #train_step = tf.train.AdagradOptimizer(1e-5).minimize(cross_entropy, name='train_step')
        #train_step = tf.train.AdadeltaOptimizer(1e-1).minimize(cross_entropy, name='train_step')
        return(train_step)

    #################################################
    # 学習を行う
    # [Args]:
    #    images     :画像のリスト
    #    labels     :ラベルのリスト
    #    steps      :学習ステップ
    #    batch_size :バッチサイズ
    #    save_step  :モデルを保存する周期
    #    start_step :学習開始のステップ
    #################################################
    def train(self, images, labels, steps, batch_size, save_step, start_step):

        # 正解率
        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # データ数取得
        num_data = len(images)

        # 指定したステップ数の学習を行う
        for step in range(start_step, steps):

            # アクセスする画像はランダムにする。
            random_i = range(num_data)
            random.shuffle(random_i)

            # 1ステップで全画像を1回学習させる。
            num_loop = num_data / batch_size
            for count in range(num_loop):
                offset = count * batch_size
                train_image_batch = []
                train_label_batch = []

                # 1度の学習でbatch_sizeの画像を使用する。
                for i in range(batch_size):
                    train_image_batch.append(images[random_i[i + offset]])
                    train_label_batch.append(labels[random_i[i + offset]])

                # 計算実行
                self.session.run(self.train_step,
                                 feed_dict={self.x: train_image_batch,
                                            self.y_: train_label_batch,
                                            self.keep_prob: RATE_DROPOUT})

                # 正解率表示
                if (step % STEP_PRINT_ACC == 0) and (count == 0):
                    feed_dict = {self.x: train_image_batch, self.y_: train_label_batch, self.keep_prob: 1.0}
                    train_accuracy, loss = self.session.run([accuracy, self.cross_entropy], feed_dict=feed_dict)
                    print("step %d, training accuracy %g, loss %f" % (step, train_accuracy, loss))

            # 指定したstep毎にモデルを保存する。
            if (step % save_step == 0) and (step != 0):
                print("step %d, save model" % step)
                self.save(step)

    #################################################
    # テストを行う
    # [Args]:
    #    images :画像のリスト
    #    labels :ラベルのリスト
    # [Returns]:
    #    test_accuracy :テストの正解率
    #################################################
    def test(self, images, labels):

        # 正解率
        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        num_data = len(images)
        i = 0
        batch_size = 10
        test_accuracy = 0
        while i < num_data:

            test_images_batch = images[i:i + batch_size]
            test_labels_batch = labels[i:i + batch_size]

            acc = self.session.run(accuracy,
                                   feed_dict={self.x: test_images_batch,
                                              self.y_: test_labels_batch,
                                              self.keep_prob: 1.0})
            test_accuracy += acc * batch_size

            # ループカウンタ更新
            i += batch_size
            if (i + batch_size) > num_data:
                batch_size = num_data - i

        test_accuracy /= float(num_data)

        return(test_accuracy)

    #################################################
    # 各入力画像をモデルにかけた結果(スコア)を取得する。
    # [Args]:
    #    images :画像のリスト
    # [Returns]:
    #    scores :スコアのリスト
    #################################################
    def get_scores(self, images):

        num_data = len(images)

        # ラベルの作成
        # 実際には使わないので中身の数値はオールゼロ
        labels = [[0 for j in range(self.num_classes)] for i in range(num_data)]

        scores = []
        for i in range(num_data):

            score = self.session.run(self.y_conv,
                                     feed_dict={self.x: [images[i]],
                                                self.y_: [labels[i]],
                                                self.keep_prob: 1.0})
            scores.append(score[0])


        return(scores)
