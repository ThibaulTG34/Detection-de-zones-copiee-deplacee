from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QToolButton, QAction, QTreeView
from PyQt5.QtGui import QPixmap, QImage, QColor, QIcon
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt,QObject
import sys
import cv2
import numpy as np
import KeyPointDetection as kpd
import os

class SourceTree(QTreeView):
    def __init__(self, main, parent=None):
        super(SourceTree, self).__init__(parent)
        self.main = main
        self.setDragEnabled(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)
        self.setDefaultDropAction(QtCore.Qt.CopyAction)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
 
 
    def set_source(self, folder):
        self.up_folder = os.path.dirname(folder)
        self.dirModel = QtGui.QFileSystemModel()
        self.dirModel.setRootPath(QtCore.QDir.homePath())
        self.setModel(self.dirModel)
        self.setRootIndex(self.dirModel.index(self.up_folder)) 
        self.setWordWrap(True)
        self.hideColumn(1)
        self.hideColumn(2)
        self.hideColumn(3)
        idx = self.dirModel.index(folder)
        self.expand(idx)
        #FIXME the following line don't works on PyQt 4.7.0
        self.scrollTo(idx, QtGui.QAbstractItemView.EnsureVisible)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt static label demo")
        self.disply_width = 1280
        self.display_height = 720
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        
        # create a text label
        self.textLabel = QLabel('Image')

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
    
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        self.button = QToolButton()
        self.button.setText("button")
        icon = QIcon()
        icon.addFile("left.svg")
        self.button.setIcon(icon)
        vbox.addWidget(self.button)
        
        cv_img1 = cv2.imread("CoMoFoD_small_v2/001_O.png")
        cv_img2 = cv2.imread("CoMoFoD_small_v2/001_F.png")
        multi = np.concatenate((cv_img1, cv_img2), axis=1)
        
        kp = kpd.KeyPoint()
        self.button.clicked.connect(lambda : kp.KeyPointDetector(cv_img1, cv_img2))
        if kp.display:
            print("if")
            qt_img2 = self.convert_cv_qt(kp.KPimg)
            self.image_label.setPixmap(qt_img2)
        else:
            print(kp.display)
            qt_img1 = self.convert_cv_qt(multi)
            self.image_label.setPixmap(qt_img1)

    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    tree = SourceTree(app)
    a.show()
    sys.exit(app.exec_())