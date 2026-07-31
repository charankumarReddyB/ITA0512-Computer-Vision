#32
import numpy as np
import cv2
import matplotlib.pyplot as plt

s = int(input("Enter image size: "))

img = np.ones((s,s,3),dtype=np.uint8)*255

b = s//10

img[0:b,0:b] = [0,0,0]
img[0:b,s-b:s] = [255,0,0]
img[s-b:s,0:b] = [0,255,0]
img[s-b:s,s-b:s] = [0,0,255]

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
