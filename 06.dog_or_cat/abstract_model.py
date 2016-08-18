# -*- coding: utf-8 -*-

# AbstractModel.py

# import
import os
from abc import ABCMeta, abstractmethod
import tensorflow as tf
import model_saver

class AbstractModel(object):

    __metaclass__ = ABCMeta

    #################################################
    # Variableを初期化する
    #################################################
    @abstractmethod
    def initialize_variables(self):
        pass

    #################################################
    # ファイルに読み書きするVariableを取得する。
    #################################################
    @abstractmethod
    def get_saveable_variables(self):
        pass

    #################################################
    # クロスエントロピーを返す
    # [Returns]:
    #    cross_entropy:クロスエントロピー(誤差関数)
    #################################################
    @abstractmethod
    def get_cross_entropy(self):
        pass

    #################################################
    # Train Stepを返す
    # [Args]:
    #    cross_entropy:
    # [Returns]:
    #    train_step:
    #################################################
    @abstractmethod
    def get_train_step(self, cross_entropy):
        pass

    #################################################
    # コンストラクタ
    # [Args]:
    #    width           :入力画像の幅
    #    height          :入力画像の高さ
    #    color_channels  :色の数(RGBなら3,白黒なら1)
    #    num_classes     :正解(ラベル)の数(種類)
    #    checkpoint      :チェックポイントのファイルパス
    #################################################
    def __init__(self, width, height, color_channels, num_classes, checkpoint):
        self.image_width = width
        self.image_height = height
        self.color_channels = color_channels
        self.num_classes = num_classes
        self.checkpoint = checkpoint

    #################################################
    # モデルの構築
    #################################################
    def construct(self):

        # Variableを初期化する
        self.initialize_variables()

        # 誤差関数=クロスエントロピー
        self.cross_entropy = self.get_cross_entropy()

        # 学習ステップ生成
        self.train_step = self.get_train_step(self.cross_entropy)

        # セッション生成
        self.session = tf.Session()

        # Variableの初期化
        init = tf.initialize_all_variables()
        self.session.run(init)

    #################################################
    # モデルのロード
    #################################################
    def load(self):

        # Variableの初期化
        self.initialize_variables()

        # セッション生成
        self.session = tf.Session()

        # 保存したモデルを読み込む。
        checkpoint_dir = os.path.dirname(self.checkpoint)
        checkpoint_file = os.path.basename(self.checkpoint)
        model_saver.restore_model(self.session, checkpoint_dir, checkpoint_file)

    #################################################
    # 学習の準備
    #################################################
    def ready_for_train(self):

        # 誤差関数=クロスエントロピー
        self.cross_entropy = self.get_cross_entropy()

        # 学習ステップ生成
        self.train_step = self.get_train_step(self.cross_entropy)

    #################################################
    # モデルを保存する
    # [Args]:
    #    global_step :学習ステップ(途中経過を保存する際に指定する)
    #################################################
    def save(self, global_step=0):
        variables = self.get_saveable_variables()
        checkpoint_dir = os.path.dirname(self.checkpoint)
        checkpoint_file = os.path.basename(self.checkpoint)
        model_saver.save_model(
            self.session, variables, checkpoint_dir, checkpoint_file, global_step)

