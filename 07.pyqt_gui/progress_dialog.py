# -*- coding: utf-8 -*-

# progress_dialog.py

# import
import subprocess
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import (QDialog, QLabel, QProgressBar, QSizePolicy, QVBoxLayout, QMessageBox)

# Constant
DEMO_EXE_PATH = '../06.dog_or_cat/main.py demo'
MAX_RANGE = 20

#################################################
# 画面識別スレッド
#################################################
class IdentifyThread(QThread):

    sig_status = QtCore.pyqtSignal(long)

    def __init__(self, imagePath, ckptPath, resultPath, parent=None):
        super(IdentifyThread, self).__init__(parent)
        self.imagePath = imagePath
        self.ckptPath = ckptPath
        self.resultPath = resultPath
        self.stopped = False
        self.mutex = QtCore.QMutex()

    def run(self):
        cmd = 'python ' + DEMO_EXE_PATH + ' ' + self.ckptPath + ' ' + self.imagePath + ' ' + self.resultPath

        # 識別実行
        try:
            status = subprocess.check_call(cmd.split(' '))
        except subprocess.CalledProcessError, e:
            status = e.returncode

        if status != 0:
            reply = QMessageBox.warning(self, 'Error', '画像識別に失敗しました')

        # 終了を通知
        self.finished.emit()


#################################################
# 画像識別進捗画面
#################################################
class ProgressDialog(QDialog):

    signal = pyqtSignal()

    #################################################
    # コンストラクタ
    #################################################
    def __init__(self, imagePath, ckptPath, resultPath, parent=None):
        super(ProgressDialog, self).__init__(parent)
        self.imagePath = imagePath
        self.ckptPath = ckptPath
        self.resultPath = resultPath
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

        self.setWindowTitle('画像識別')
        self.setLayout(layout)

    #################################################
    # 表示
    #################################################
    def show(self):

        # Timer 200ミリ秒周期で5%進捗を進める
        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.timerEvent)
        self.timer.start()

        # 初期値は0%
        self.step = 0
        self.prgsBar.setValue(self.step)

        # 識別開始
        self.thread = IdentifyThread(self.imagePath, self.ckptPath, self.resultPath)
        self.thread.finished.connect(self.identify_finished)
        self.thread.start()

        # ダイアログ表示
        super(ProgressDialog, self).show()

    #################################################
    # 識別終了イベント
    #################################################
    def identify_finished(self):
        if (self.thread is not None):
            self.thread.wait()
        self.thread = None
        self.signal.emit()
        self.close()

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
        super(ProgressDialog, self).close()

