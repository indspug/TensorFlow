# -*- coding: utf-8 -*-

# selec_window.py

# import
import os
import sys
import subprocess
import PyQt5
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QDialog, QPushButton, QLabel, \
                            QGridLayout, QHBoxLayout, QVBoxLayout, QSizePolicy
from PyQt5.QtGui import QPixmap, QFont

# Constant
DOG_IMAGE = './typical/dog.jpg'
CAT_IMAGE = './typical/cat.jpg'
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

#################################################
# 結果画面
#################################################
class ResultWindow(QDialog):

    #################################################
    # コンストラクタ
    #################################################
    def __init__(self, score):
        super(ResultWindow, self).__init__()
        self.score = score
        self.init_widgets()

    #################################################
    # Widgetsの初期化
    #################################################
    def init_widgets(self):

        layout = QGridLayout()

        print(self.score)

        # String -> Float
        catScore = float(self.score[0])*100
        dogScore = float(self.score[1])*100

        # Result Labels
        resultLayout = QHBoxLayout()
        resultLabel1 = QLabel('この画像は')
        resultLabel1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        if( catScore > dogScore ):
            resultLabel2 = QLabel('猫')
        else:
            resultLabel2 = QLabel('犬')
        resultLabel2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        font = resultLabel2.font()
        font.setPointSize(24)
        resultLabel2.setFont(font)
        resultLabel3 = QLabel('ですね？')
        resultLabel3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        resultLayout.addWidget(resultLabel1)
        resultLayout.addWidget(resultLabel2)
        resultLayout.addWidget(resultLabel3)
        resultLayout.addStretch(10)
        layout.addLayout(resultLayout, 0, 0)

        # Tipycal Image Label
        tipImgLabel = QLabel()
        if( catScore > dogScore ):
            pixmap = QPixmap(CAT_IMAGE)
        else:
            pixmap = QPixmap(DOG_IMAGE)
        pixmap = pixmap.scaled(IMAGE_WIDTH, IMAGE_HEIGHT, QtCore.Qt.KeepAspectRatio)
        tipImgLabel.setPixmap(pixmap)
        layout.addWidget(tipImgLabel, 1, 0)

        # Score Labels
        scoreLayout = QVBoxLayout()
        dogScoreLabel = QLabel('cat={0:.1f}%'.format(catScore))
        catScoreLabel = QLabel('dog={0:.1f}%'.format(dogScore))
        dogScoreLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        catScoreLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        scoreLayout.addWidget(dogScoreLabel)
        scoreLayout.addWidget(catScoreLabel)
        scoreLayout.addStretch(10)
        layout.addLayout(scoreLayout, 1, 1, 1, 1)

        # Close Button
        closeButton = QPushButton('閉じる')
        closeButton.clicked.connect(self.close)
        layout.addWidget(closeButton, 2, 1)

        # Title
        self.setWindowTitle('画像識別')

        self.setLayout(layout)
