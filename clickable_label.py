from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import pyqtSignal, Qt


class ClickableLabel(QLabel):
    clicked = pyqtSignal()  # QLabel엔 없던 시그널을 새로 정의

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()  # 클릭되면 clicked 신호 발사!
        super().mousePressEvent(event)
