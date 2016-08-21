# -*- coding: utf-8 -*-

# image_select_box.py

# import
import os
from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QLabel, QComboBox, QSizePolicy
from PyQt5.QtGui import QPixmap

# Constant
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

#################################################
# 画像選択ボックス
#################################################
class ImageSelectBox(QWidget):

    #################################################
    # コンストラクタ
    # [Args]:
    #    image_dir:表示する画像が格納されているディレクトリ
    #################################################
    def __init__(self, image_dir):
        super(ImageSelectBox, self).__init__()
        self.image_dir = image_dir
        self.image_files = []
        self.init_widgets()

    #################################################
    # Widgetsの初期化
    #################################################
    def init_widgets(self):

        # Group Box
        self.groupBox = QGroupBox('画像選択')
        boxLayout = QVBoxLayout()
        self.groupBox.setLayout(boxLayout)

        # Message Label
        msgLabel = QLabel('識別する画像を選択してください')
        msgLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        boxLayout.addWidget(msgLabel)

        # Combo Box
        self.comboBox = QComboBox(self)
        self.comboBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        boxLayout.addWidget(self.comboBox)
        self.comboBox.activated[str].connect(self.comboActivated)

        # Image Label
        self.imgLabel = QLabel(self)
        boxLayout.addWidget(self.imgLabel)

        layout = QVBoxLayout()
        layout.addWidget(self.groupBox)
        self.setLayout(layout)

    #################################################
    # 表示の更新
    #################################################
    def update(self):

        # ディレクトリに新しいファイルが追加されたら、リストにファイル名を追加する
        files = os.listdir(self.image_dir)
        for filepath in files:
            filename = os.path.basename(filepath)
            if not (filename in self.image_files):
                self.image_files.append(filename)

        # リストに登録されているファイルが削除されたら、リストからファイル名を削除する
        del_filenames = []
        for filename in self.image_files:
            filepath = os.path.join(self.image_dir, filename)
            if not (os.path.exists(filepath)):
                del_filenames.append(filename)

        for filename in del_filenames:
            self.image_files.remove(filename)

        # 昇順でソート
        self.image_files.sort()

        # コンボボックスの表示を更新
        self.comboBox.clear()
        for filename in self.image_files:
            self.comboBox.addItem(filename)

        # インデックス初期設定
        self.comboActivated(self.comboBox.currentText())

    #################################################
    # 画像ファイル名を取得する。
    #################################################
    def get_image_name(self):
        text = self.comboBox.currentText()
        return(text)

    #################################################
    # Combo Boxのイベント
    #################################################
    def comboActivated(self, text):

        # 選択した画像を表示する
        filepath = os.path.join(self.image_dir, text)
        if(os.path.isfile(filepath)):
            pixmap = QPixmap(filepath)
            pixmap = pixmap.scaled(IMAGE_WIDTH, IMAGE_HEIGHT, QtCore.Qt.KeepAspectRatio)
            self.imgLabel.setPixmap(pixmap)
