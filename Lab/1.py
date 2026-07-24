#1
from google.colab import files
import cv2
from matplotlib import pyplot as plt
uploaded = files.upload()
import os
file_name = list(uploaded.keys())[0]
img = cv2.imread(file_name)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.title("Gray Image")
plt.axis('off')
plt.show()
