from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import cv2
import numpy as np
from ForgeryDetection import ForgeryDetection
from PyQt5 import QtWidgets, QtGui, QtCore
import keras
from keras.models import load_model

IMG_SIZE_OCV = 512
IMG_SIZE_CNN = 224
SIZE_WIDGET = 720
MODEL_NAME = "modelForgery.h5"
HEIGHT_INFO_IMAGE = 720
WIDTH_INFO_IMAGE = 50

class MainWindowApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Detector2000")

        self.splitter = QSplitter(self)
        self.setCentralWidget(self.splitter)

        self.sift_detector = App()
        self.CNN_detector = CNNApp()

        self.sift_detector.setMinimumSize(SIZE_WIDGET, SIZE_WIDGET)
        self.CNN_detector.setMinimumSize(SIZE_WIDGET, SIZE_WIDGET)

        self.splitter.addWidget(self.sift_detector)
        self.splitter.addWidget(self.CNN_detector)

        # self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        # self.setStyleSheet('background-color: lightblack')


class CNNApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image_label = QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        
        self.loadButton = QPushButton('Charger les images', self)
        self.loadButton.clicked.connect(self.LoadImages)

        self.cnnButton = QPushButton("Use CNN", self)
        self.cnnButton.clicked.connect(self.Predict)
        self.cnnButton.setVisible(False)

        self.image_info = QLabel(self)
        self.image_info.setMaximumSize(HEIGHT_INFO_IMAGE, WIDTH_INFO_IMAGE)
        self.image_info.setAlignment(QtCore.Qt.AlignCenter)

        self.cnn_result = QLabel(self)
        self.cnn_result.setMaximumSize(HEIGHT_INFO_IMAGE, WIDTH_INFO_IMAGE)
        myFont=QtGui.QFont()
        myFont.setBold(True)
        self.cnn_result.setFont(myFont)
        self.cnn_result.setAlignment(QtCore.Qt.AlignCenter)
        
        self.CNNPart = QLabel(self)
        self.CNNPart.setMaximumSize(HEIGHT_INFO_IMAGE, WIDTH_INFO_IMAGE)
        self.CNNPart.resize(HEIGHT_INFO_IMAGE, WIDTH_INFO_IMAGE)
        self.CNNPart.setAlignment(QtCore.Qt.AlignCenter)
        
        self.CNNPart.move(130, 20)
        self.CNNPart.setFont(myFont)
        self.CNNPart.setText("Méthode par Apprentissage profond")

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.cnn_result)
        layout.addWidget(self.image_info)
        layout.addWidget(self.cnnButton)
        layout.addWidget(self.loadButton)
        self.setLayout(layout)

        self._load_model = load_model(MODEL_NAME)

        self.images = []
        self.imagesOCV = []
        self.current_index = 0
        self.image_filename = []

        self.image_info.setAttribute(QtCore.Qt.WA_StyledBackground, True)

    def LoadImages(self):
        file_names, _ = QFileDialog.getOpenFileNames(self, 'Ouvrir une image', '', 'Images (*.png *.xpm *.jpeg *.bmp *.tif *.jpg)')

        if(len(file_names) > 0):
            self.cnnButton.setVisible(True)
            self.cnn_result.setText("")
            self.images = []
            self.imagesOCV = []
            self.current_index = 0
            self.image_filename = []
            self.image_info.setStyleSheet('background-color: None')

            for file_name in file_names:
                image = cv2.imread(file_name)
                image = cv2.resize(image, (IMG_SIZE_OCV,IMG_SIZE_OCV))
                self.imagesOCV.append(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                self.image_filename.append(file_name)
                self.images.append(qimage)
                self.DisplayImages()

    def DisplayImages(self):
        if self.images:
            pixmap = QPixmap.fromImage(self.images[self.current_index])
            self.image_info.setText("Image: " + self.image_filename[self.current_index] + " - Taille: " + str(self.imagesOCV[self.current_index].shape))
            self.image_label.setPixmap(pixmap)

    def Predict(self):
        img_resized = cv2.resize(self.imagesOCV[self.current_index], (IMG_SIZE_CNN, IMG_SIZE_CNN))
        h,w,_ = img_resized.shape
        img_resized.shape = (-1, h, w, 3)
        
        img_resized = img_resized.astype('float')
        img_resized = img_resized/255.0
        prediction = self._load_model.predict(img_resized)
        print(prediction)
        if(prediction[0][0]>=0.5):
            self.image_info.setStyleSheet('background-color: darkgreen')
        else:
            self.image_info.setStyleSheet('background-color: darkred')
        self.cnn_result.setText("Original: " + str(round(prediction[0][0]*100,2)) + "% - Forgery: " + str(round(prediction[0][1]*100,2)) + "%")
            

class App(QWidget):
    def __init__(self):
        super().__init__()

        self.image_label = QLabel(self)
        #self.image_label.setMinimumSize(512, 512)
        #self.image_label.setGeometry(50, 150, 512, 512)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.loadButton = QPushButton('Charger les images', self)
        #self.loadButton.setGeometry(QtCore.QRect(10, 10, 100, 30))
        self.loadButton.clicked.connect(self.convert_cv_qt)
        
        self.image_info = QLabel(self)
        self.image_info.setMaximumSize(HEIGHT_INFO_IMAGE, WIDTH_INFO_IMAGE)
        self.image_info.setAlignment(QtCore.Qt.AlignCenter)
        
        self.forgery_info = QLabel(self)
        self.forgery_info.setMaximumSize(HEIGHT_INFO_IMAGE, WIDTH_INFO_IMAGE)
        self.forgery_info.setAlignment(QtCore.Qt.AlignCenter)
        self.forgery_info.setText("")
        
        self.forgeryButton = QPushButton('Détection de la falsification', self)
        self.forgeryButton.clicked.connect(self.forgeryDetection)
        
        self.DBSCAN_Part = QLabel(self)
        self.DBSCAN_Part.setMaximumSize(HEIGHT_INFO_IMAGE, WIDTH_INFO_IMAGE)
        self.DBSCAN_Part.resize(HEIGHT_INFO_IMAGE, WIDTH_INFO_IMAGE)
        self.DBSCAN_Part.setAlignment(QtCore.Qt.AlignCenter)
        
        self.DBSCAN_Part.move(130, 20)
        myFont=QtGui.QFont()
        myFont.setBold(True)
        self.DBSCAN_Part.setFont(myFont)
        self.DBSCAN_Part.setText("Méthode par clustering")
        
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setFocus()
        
        self.detectButton = QPushButton("Détection de Points d'intérêt", self)
        self.detectButton.clicked.connect(self.keypointsDetection)
        
        self.forgeryButton.setVisible(False)
        self.detectButton.setVisible(False)
       
        
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.image_info)
        layout.addWidget(self.forgery_info)
        layout.addWidget(self.forgeryButton)
        layout.addWidget(self.detectButton)
        layout.addWidget(self.loadButton)
        self.setLayout(layout)

        
        self.images = []
        self.imagesOCV = []
        self.current_index = 0
        self.image_filename = []
    
            
    def keypointsDetection(self):
        forgery_detector = ForgeryDetection(self.imagesOCV[self.current_index])
        img_orig, img, filename = forgery_detector.KeyPointDetector(self.imagesOCV[self.current_index], self.image_filename[self.current_index])
        
        if img is not None:
            self.forgeryButton.setVisible(True)    
            qimage = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_BGR888)
            self.images.append(qimage)
            self.image_filename.append(filename)
            self.imagesOCV.append(img_orig)
            self.imagesOCV.append(img)
            self.current_index += 1
            self.DisplayImages()    
            
    
    def forgeryDetection(self):
        forgery_detector = ForgeryDetection(self.imagesOCV[self.current_index])
        img, name = forgery_detector.ForgeryDetect(self.imagesOCV[self.current_index], self.image_filename[self.current_index])
        
        if img is not None:
            qimage = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_BGR888)
            self.images.append(qimage)
            self.image_filename.append(name)
            self.imagesOCV.append(img)
            self.current_index += 1
            self.DisplayImages() 
        else:
            self.forgery_info.setText(self.image_filename[self.current_index] + " n'est pas falsifiée !")
        
    
            
    def convert_cv_qt(self):
        file_names, _ = QFileDialog.getOpenFileNames(self, 'Ouvrir une image', '', 'Images (*.png *.xpm *.jpeg *.bmp *.tif *.jpg)')
        self.forgery_info.setText("")
        if(len(file_names) > 0):
            self.detectButton.setVisible(True)
            self.forgeryButton.setVisible(False)
            self.images = []
            self.imagesOCV = []
            self.current_index = 0
            self.image_filename = []
        
            for file_name in file_names:
                image = cv2.imread(file_name)
                image = cv2.resize(image, (IMG_SIZE_OCV,IMG_SIZE_OCV))
                self.imagesOCV.append(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                self.image_filename.append(file_name)
                self.images.append(qimage)
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
    a = MainWindowApp()
    # a.show()
    a.showMaximized()
    sys.exit(app.exec_())