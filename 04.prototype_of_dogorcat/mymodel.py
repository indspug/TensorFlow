# -*- coding: utf-8 -*-

# インポート
import tensorflow as tf
import model_saver
import random

FILTER_SIZE1 = 3
FILTER_SIZE2 = 3
LAST_FILTER_NUM = 64
WIDTH_RATIO = 4
HEIGHT_RATIO = 4

class mymodel(object):


    #################################################
    # コンストラクタ
    #################################################
    def __init__(self, width, height, color_channels, num_classes, directory, filename):
        self.image_width = width                # 入力画像の幅
        self.image_height = height              # 入力画像の高さ
        self.color_channels = color_channels    # 色の数(RGBなら3,白黒なら1)
        self.num_classes = num_classes          # 正解の種類
        self.save_directory = directory         # モデル保存先のディレクトリ
        self.save_filename = filename           # モデル保存ファイル名

    #################################################
    # 初期化された重みVariableを返す
    #################################################
    def weight_variable(self, shape, name=None):
        # 正規分布に従うランダム値を返す。stddevは標準偏差
        initial = tf.truncated_normal(shape, stddev=0.1)
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
    # ファイルに読み書きするVariableを初期化する
    #################################################
    def initialize_io_variables(self):

        # First Convolutional Layer
        self.W_conv1 = self.weight_variable([FILTER_SIZE1, FILTER_SIZE1, self.color_channels, 32], name='W_conv1')
        self.b_conv1 = self.bias_variable([32], name='b_conv1')

        # Second Convolutional Layer
        self.W_conv2 = self.weight_variable([FILTER_SIZE2, FILTER_SIZE2, 32, 64], name='W_conv2')
        self.b_conv2 = self.bias_variable([64], name='b_conv2')

        # Densely Connected Layer(全結合層)
        # 入力を3次元から1次元のテンソルに変更
        self.W_fc1 = self.weight_variable([self.image_width/WIDTH_RATIO * self.image_height/HEIGHT_RATIO * LAST_FILTER_NUM, 1024], name='W_fc1')
        self.b_fc1 = self.bias_variable([1024], name='b_fc1')

        # Readout Leyer(出力層)
        self.W_fc2 = self.weight_variable([1024, self.num_classes], name='W_fc2')
        self.b_fc2 = self.bias_variable([self.num_classes], name='b_fc2')

        # ドロップアウトを使用する場合のドロップアウト率。
        self.keep_prob = tf.placeholder(tf.float32)

    #################################################
    # 計算途中で使用するVariableを初期化する
    #################################################
    def initialize_temporary_variables(self):

        # 入力データ格納用Variable作成
        num_elm_x = self.image_width * self.image_height * self.color_channels
        self.x = tf.placeholder(tf.float32, [None, num_elm_x])

        # 正解データ格納用Variable作成
        self.y_ = tf.placeholder(tf.float32, [None, self.num_classes])

        # First Convolutional Layerの入力
        # 1d dimensionは画像数(-1は元サイズと同じの意味)
        # 2,3d dimensionは画像のwidth,heightに対応
        # 4d dimensionはcolor channlesに対応
        print('image_width : %d' % self.image_width)
        print('image_height : %d' % self.image_height)
        print('color_channels : %d' % self.color_channels)
        x_image = tf.reshape(self.x, [-1, self.image_width, self.image_height, self.color_channels])

        # First Convolutional Layerの計算
        h_conv1 = tf.nn.relu(self.conv2d(x_image, self.W_conv1) + self.b_conv1)
        h_pool1 = self.max_pool(h_conv1)

        # h_pool1にドロップアウトを適用する。
        h_pool1_drop = tf.nn.dropout(h_pool1, self.keep_prob)

        # Second Convolutional Layerの計算
        #h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.W_conv2) + self.b_conv2)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1_drop, self.W_conv2) + self.b_conv2)
        h_pool2 = self.max_pool(h_conv2)

        # h_pool2にドロップアウトを適用する。
        h_pool2_drop = tf.nn.dropout(h_pool2, self.keep_prob)

        # Second Convolutional Layerの出力を
        # 入力を3次元から1次元のテンソルに変更
        #h_pool2_flat = tf.reshape(h_pool2, [-1, self.image_width/WIDTH_RATIO * self.image_height/HEIGHT_RATIO * LAST_FILTER_NUM])
        h_pool2_flat = tf.reshape(h_pool2_drop, [-1, self.image_width / WIDTH_RATIO * self.image_height / HEIGHT_RATIO * LAST_FILTER_NUM])

        # h_pool2_flatとW_fc1を乗算し、バイアスを加算した値に
        # ReLUを適用する。
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat , self.W_fc1) + self.b_fc1)

        # h_fc1に対してドロップアウトを適用する。
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Readout Leyer(出力層)
        self.y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, self.W_fc2) + self.b_fc2)


    #################################################
    # モデルを構築する
    #################################################
    def construct(self):

        #Variableを初期化する
        self.initialize_io_variables()
        self.initialize_temporary_variables()

        # 誤差関数=クロスエントロピー
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv), reduction_indices=[1]))

        # Adam Algorithmで学習係数=0.00001
        self.train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

        # セッション生成
        self.session = tf.Session()
        init = tf.initialize_all_variables()
        self.session.run(init)

    #################################################
    # モデルを保存する
    #################################################
    def save(self, global_step=0):
        variables = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2,
                     self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]
        model_saver.save_model(
            self.session, variables, self.save_directory, self.save_filename, global_step)

    #################################################
    # モデルをロードする
    #################################################
    def load(self, directory, filename):

        # Variableの初期化
        self.initialize_io_variables()

        # セッション生成
        self.session = tf.Session()

        # 保存したモデルを読み込む。
        model_saver.restore_model(self.session, directory, filename)

        # Variableの初期化
        self.initialize_temporary_variables()

    #################################################
    # 学習を行う
    #################################################
    def train(self, images, labels, steps, batch_size, save_step):

        # 正解率
        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        num_data = len(images)
        print(num_data)

        # 指定したステップ数の学習を行う
    	for step in range(steps):

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
                    train_image_batch.append(images[random_i[i+offset]])
                    train_label_batch.append(labels[random_i[i+offset]])

                # Dropout率=0.5で計算
                self.session.run(self.train_step, feed_dict={self.x: train_image_batch, self.y_: train_label_batch, self.keep_prob: 0.5})

                if (step % 10 == 0) and (count == 0):
                    train_accuracy = self.session.run(self.accuracy, feed_dict={self.x: train_image_batch, self.y_: train_label_batch, self.keep_prob: 1.0})
                    #train_accuracy = self.session.run(self.accuracy, feed_dict={self.x: images, self.y_: labels, self.keep_prob: 1.0})
                    print("step %d, training accuracy %g" % (step, train_accuracy))

            # 指定したstep毎にモデルを保存する。
            if (step % save_step) == 0 :
                print("step %d, save model" % step)
                self.save(step)

    #################################################
    # テストを行う
    #################################################
    def test(self, images, labels):

        # 正解率
        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        num_data = len(images)
        i = 0
        test_accuracy = 0
        batch_size = 20
        while i < num_data:

            test_images_batch = images[i:i+batch_size]
            test_labels_batch = labels[i:i+batch_size]

            acc = self.session.run(self.accuracy, feed_dict={self.x: test_images_batch, self.y_: test_labels_batch, self.keep_prob: 1.0})
            test_accuracy += acc * batch_size

            # ループカウンタ更新
            i += batch_size
            if (i+batch_size) > num_data:
                batch_size = num_data - i

        test_accuracy /= float(num_data)
        print("test accuracy = %g" % test_accuracy)

