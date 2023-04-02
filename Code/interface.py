from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import cv2
import numpy as np
from ForgeryDetection import ForgeryDetection
from PyQt5 import QtWidgets, QtGui, QtCore


class App(QWidget):
    def __init__(self):
        super().__init__()
    
        self.image_label = QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(1280, 720)
        self.loadButton = QPushButton('Charger les images', self)
        #self.loadButton.setGeometry(QtCore.QRect(10, 10, 100, 30))
        self.loadButton.clicked.connect(self.convert_cv_qt)
        
        self.image_info = QLabel(self)
        self.image_info.setGeometry(10, 700, 700, 20)
        self.image_info.setText("Image: " + " - Taille: ")
        
        self.forgeryButton = QPushButton('Détection de la falsification', self)
        self.forgeryButton.clicked.connect(self.forgeryDetection)
        
        # Créer un bouton pour afficher le plot
        self.plotButton = QPushButton('Afficher le plot', self)
        self.plotButton.clicked.connect(self.displayPlot)
        
        
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setFocus()
        
        self.detectButton = QPushButton("Détection de Points d'intérêt", self)
        self.detectButton.clicked.connect(self.keypointsDetection)
        
        self.forgeryButton.setVisible(False)
        self.plotButton.setVisible(False)
        self.detectButton.setVisible(False)
        
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.loadButton)
        layout.addWidget(self.detectButton)
        layout.addWidget(self.forgeryButton)
        layout.addWidget(self.plotButton)
        self.setLayout(layout)

        
        self.images = []
        self.imagesOCV = []
        self.current_index = 0
        self.image_filename = []
        
    
    def displayPlot(self):
        forgery_detector = ForgeryDetection(self.imagesOCV[self.current_index])
        forgery_detector.displayPlot()

    
            
    def keypointsDetection(self):
        forgery_detector = ForgeryDetection(self.imagesOCV[self.current_index])
        img_orig, img, filename = forgery_detector.KeyPointDetector(self.imagesOCV[self.current_index], self.image_filename[self.current_index])
        
        if img is not None:
            self.forgeryButton.setVisible(True)    
            qimage = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
            self.images.append(qimage)
            self.image_filename.append(filename)
            self.imagesOCV.append(img_orig)
            self.DisplayImages()    
            
    
    def forgeryDetection(self):
        forgery_detector = ForgeryDetection(self.imagesOCV[self.current_index])
        img, name = forgery_detector.ForgeryDetect(self.imagesOCV[self.current_index], self.image_filename[self.current_index])
        
        if img is not None:
            self.plotButton.setVisible(True)
            qimage = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
            self.images.append(qimage)
            self.image_filename.append(name)
            self.imagesOCV.append(img)
            self.DisplayImages() 
        
    
            
    def convert_cv_qt(self):
        file_names, _ = QFileDialog.getOpenFileNames(self, 'Ouvrir une image', '', 'Images (*.png *.xpm *.jpg *.bmp)')
        self.detectButton.setVisible(True)
        for file_name in file_names:
            image = cv2.imread(file_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
            self.image_filename.append(file_name)
            self.images.append(qimage)
            self.imagesOCV.append(image)
            self.DisplayImages()
    
    
    def DisplayImages(self):
        if self.images:
            pixmap = QPixmap.fromImage(self.images[self.current_index])
            self.image_info.setText("Image: " + self.image_filename[self.current_index] + " - Taille: " + str(self.imagesOCV[self.current_index].shape))
            self.image_label.setPixmap(pixmap)
    
    
    def keyPressEvent(self, event):
        
        # Permet de naviguer entre les images avec les touches gauche et droite du clavier
        print(self.current_index)
        if event.key() == Qt.Key.Key_Left:
            
            if len(self.images) > 1:
                self.current_index = (self.current_index - 1) % len(self.images)
                self.DisplayImages()
                
        elif event.key() == Qt.Key.Key_Right:

            if len(self.images) > 1:
                self.current_index = (self.current_index + 1) % len(self.images)
                self.DisplayImages()
        
    
if __name__=="__main__":
    app = QApplication([])
    a = App()
    a.show()
    sys.exit(app.exec_())