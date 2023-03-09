import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

def loadDataset():
    _pathDataset = 'CoMoFoD_small_v2'
    _imagesPath = [ f for f in listdir(_pathDataset) if isfile(join(_pathDataset,f)) ]
    _images = np.empty(len(_imagesPath), dtype=object)
    for index in range(0, len(_imagesPath)):
        print(join(_pathDataset,_imagesPath[index]))
        _images[index] = cv2.imread( join(_pathDataset,_imagesPath[index]) )

    START_INDEX = 25
    NB_PLOT_IMG = 25

    #imPlot(_images, START_INDEX, NB_PLOT_IMG)

def imPlot(_images, _start, _nb):
    plt.figure(figsize=(15,15))
    for i in range(_start,_start+_nb):
        print(i)
        plt.subplot(5,5,i%_nb+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        _images[i] = cv2.cvtColor(_images[i], cv2.COLOR_BGR2RGB)
        plt.imshow(_images[i],cmap=plt.cm.binary)  
        plt.xlabel('taille ' + str(_images[i].shape))
    plt.show()

def create_training_data(path_data, list_classes):
  training_data=[]
  for classes in list_classes:
      path=os.path.join(path_data, classes)
      class_num=list_classes.index(classes)
      for img in os.listdir(path):
        try:
          img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_UNCHANGED)
          new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
          training_data.append([new_array, class_num])
        except Exception as e:
          pass  
  return training_data    

def create_X_y (path_data, list_classes):
      # récupération des données
      training_data=create_training_data(path_data, list_classes)
      # tri des données
      random.shuffle(training_data)
      # création de X et y
      X=[]
      y=[]
      for features, label in training_data:
        X.append(features)
        y.append(label)
      X=np.array(X).reshape(-1,IMG_SIZE, IMG_SIZE, 3)
      y=np.array(y)
      return X,y

def plot_examples(X,y):
  plt.figure(figsize=(15,15))
  for i in range(COLUMNS):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # cv2 lit met les images en BGR et matplotlib lit du RGB
    X[i] = cv2.cvtColor(X[i], cv2.COLOR_BGR2RGB)
    plt.imshow(X[i]/255.,cmap=plt.cm.binary)
    plt.xlabel('classe ' + str(y[i]))

def main() -> int:
    loadDataset()

if __name__ == '__main__':
    main()