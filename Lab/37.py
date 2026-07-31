#37
from google.colab import files
import cv2
import matplotlib.pyplot as plt
f = list(files.upload().keys())[0]
img = cv2.imread(f)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,mask = cv2.threshold(gray,120,255,cv2.THRESH_BINARY_INV)
plt.imshow(mask,cmap='gray')
plt.axis('off')
plt.show()
