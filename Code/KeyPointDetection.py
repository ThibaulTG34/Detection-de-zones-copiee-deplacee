import cv2
import numpy as np
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt

Original_img = cv2.imread("CoMoFoD_small_v2/001_O_BC1.png")
Forged_img = cv2.imread("CoMoFoD_small_v2/001_F_BC1.png")
#cv2.imshow("Original Image", Original_img)
#cv2.imshow("Forged Image", Forged_img)


img = cv2.cvtColor(Original_img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

test_img = cv2.pyrDown(img)
test_img = cv2.pyrDown(test_img)

rows, cols = test_img.shape[:2]

rot_mat = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 1)
test_img = cv2.warpAffine(test_img, rot_mat, (cols, rows))

test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

fx, plot = plt.subplots(1, 2, figsize=(20,10))

cv2.imshow("Training Image", img)
cv2.imshow("Testing Image", test_img)
    


cv2.waitKey(0)
cv2.destroyAllWindows()

