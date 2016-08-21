# -*- coding: utf-8 -*-

# checkpoint_select_box.py

# import
import os
from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QLabel, QComboBox, QSizePolicy
from PyQt5.QtGui import QPixmap

#################################################
# チェックポイント選択ボックス
#################################################
class CheckpointSelectBox(QWidget):

    #################################################
    # コンストラクタ
    # [Args]:
    #    checkpoin_dir:学習済みモデルが格納されているディレクトリ
    #################################################
    def __init__(self, checkpoin_dir):
        super(CheckpointSelectBox, self).__init__()
        self.checkpoin_dir = checkpoin_dir
        self.init_widgets()

    #################################################
    # Widgetsの初期化
    #################################################
    def init_widgets(self):

        # Group Box
        self.groupBox = QGroupBox('チェックポイント選択')
        boxLayout = QVBoxLayout()
        self.groupBox.setLayout(boxLayout)

        # Message Label
        msgLabel = QLabel('チェックポイントを選択してください。')
        msgLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        boxLayout.addWidget(msgLabel)

        # Combo Box
        self.comboBox = QComboBox(self)
        self.comboBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        boxLayout.addWidget(self.comboBox)

        layout = QVBoxLayout()
        layout.addWidget(self.groupBox)
        self.setLayout(layout)

    #################################################
    # 表示の更新
    #################################################
    def update(self):

        # コンボボックス初期化
        self.comboBox.clear()

        # ディレクトリ内にあるファイル名を追加する
        files = os.listdir(self.checkpoin_dir)
        files.sort()
        for filepath in files:
            filename = os.path.basename(filepath)
            self.comboBox.addItem(filename)

    #################################################
    # チェックポイント名を取得する。
    #################################################
    def get_checkpoint_name(self):
        text = self.comboBox.currentText()
        return(text)
