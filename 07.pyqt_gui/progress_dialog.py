# -*- coding: utf-8 -*-

# progress_dialog.py

# import
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QDialog, QLabel, QProgressBar, QSizePolicy, QVBoxLayout

# Constant
MAX_RANGE = 10

#################################################
# 画像識別進捗画面
#################################################
class ProgressDialog(QDialog):

    #################################################
    # コンストラクタ
    #################################################
    def __init__(self, parent=None):
        super(ProgressDialog, self).__init__(parent)
        self.init_widgets()

    #################################################
    # Widgetsの初期化
    #################################################
    def init_widgets(self):

        layout = QVBoxLayout()

        # Message Label
        msgLabel = QLabel('画像を識別しています。しばらくお待ちください。')
        msgLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(msgLabel)

        # Progress Bar
        self.prgsBar = QProgressBar()
        self.prgsBar.setRange(0, MAX_RANGE)
        self.prgsBar.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        layout.addWidget(self.prgsBar)

        # Title
        self.setWindowTitle('画像識別')

        self.setLayout(layout)

    #################################################
    # 表示
    #################################################
    def show(self):

        # Timer 1秒周期で10%進捗を進める
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.timerEvent)
        self.timer.start()

        # 初期値は0%
        self.step = 0
        self.prgsBar.setValue(self.step)

        super(ProgressDialog, self).show()

    #################################################
    # タイマーイベント
    #################################################
    def timerEvent(self):

        self.step += 1
        self.prgsBar.setValue(self.step)

    #################################################
    # 終了
    #################################################
    def close(self):
        print('close')

        self.timer.stop()
        super(ProgressDialog, self).close()


