# -*- coding: utf-8 -*-

# selec_window.py

# import
import os
import sys
import subprocess
import csv
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, \
                            QHBoxLayout, QVBoxLayout, QMessageBox, \
                            QDialog
from image_select_box import ImageSelectBox
from checkpoint_select_box import CheckpointSelectBox
from progress_dialog import ProgressDialog
from result_window import ResultWindow

# Constant
DEMO_EXE_PATH = '../06.dog_or_cat/main.py demo'
IMAGE_DIR = './image'
CKPT_DIR = '../06.dog_or_cat/ckpt'
RESULT_CSV = './score.csv'


#################################################
# 選択画面
#################################################
class SelectWindow(QWidget):

    #################################################
    # コンストラクタ
    #################################################
    def __init__(self):
        super(SelectWindow, self).__init__()
        self.init_widgets()

    #################################################
    # Widgetsの初期化
    #################################################
    def init_widgets(self):

        layout = QVBoxLayout()

        # Image Select Box
        self.imgSlctBox = ImageSelectBox(IMAGE_DIR)
        self.imgSlctBox.update()
        layout.addWidget(self.imgSlctBox)

        # Checkpoint Select Box
        self.ckptSlctBox = CheckpointSelectBox(CKPT_DIR)
        self.ckptSlctBox.update()
        layout.addWidget(self.ckptSlctBox)

        # Identify Button
        self.identifyButton = QPushButton('識別開始')
        self.identifyButton.clicked.connect(self.identify)

        # Close Button
        self.closeButton = QPushButton('閉じる')
        self.closeButton.clicked.connect(self.close)

        # Buttons Layout
        btnLayout = QHBoxLayout()
        btnLayout.addStretch(2)
        btnLayout.addWidget(self.identifyButton, 0)
        btnLayout.addWidget(self.closeButton, 0)
        layout.addLayout(btnLayout)

        # Title
        self.setWindowTitle('画像識別')

        self.setLayout(layout)

    #################################################
    # 識別開始ボタンクリックイベント
    #################################################
    def identify(self):

        # 進捗画面の表示が上手くいかないのでコメントアウト
        #self.dialog = ProgressDialog(self)
        #self.dialog.show()

        # 画像ファイル名とチェックポイント名取得
        image = self.imgSlctBox.get_image_name()
        imagePath = os.path.join(IMAGE_DIR, image)
        ckpt = self.ckptSlctBox.get_checkpoint_name()
        ckptPath = os.path.join(CKPT_DIR, ckpt)
        resultPath = RESULT_CSV
        cmd = 'python ' + DEMO_EXE_PATH + ' ' + ckptPath + ' ' + imagePath + ' ' + resultPath

        # 識別実行
        try:
            status = subprocess.check_call(cmd.split(' '))
        except subprocess.CalledProcessError, e:
            status = e.returncode

        if status != 0:
            reply = QMessageBox.warning(self, 'Error', '画像識別に失敗しました')

        # 進捗画面の表示が上手くいかないのでコメントアウト
        #self.dialog.close()

        # スコアをCSVから取得
        fin = open(resultPath, 'r')
        csvReader = csv.reader(fin)
        for row in csvReader:
            score = row
        fin.close()

        # 結果をダイアログ表示
        self.resultWindow = ResultWindow(score)
        self.resultWindow.show()


#################################################
# main
#################################################
if __name__ == '__main__':

    # Application
    app = QtWidgets.QApplication(sys.argv)

    # Window
    window = SelectWindow()

    # 表示
    window.show()

    sys.exit(app.exec_())

