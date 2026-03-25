from PyQt5 import QtGui
from PyQt5 import QtWidgets


class StatusStats(QtWidgets.QLabel):
    def __init__(self):
        super().__init__("")

        font = QtGui.QFont()
        font.setFamily("monospace")
        font.setStyleHint(QtGui.QFont.Monospace)
        self.setFont(font)
        
        # Add padding to create margin from edges
        self.setStyleSheet("padding: 2px 6px;")
