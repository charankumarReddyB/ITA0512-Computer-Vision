#31
from google.colab import files
import cv2
import matplotlib.pyplot as plt

f = list(files.upload().keys())[0]

img = cv2.imread(f,0)

_,th = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

plt.imshow(th,cmap='gray')
plt.axis('off')
plt.show()
