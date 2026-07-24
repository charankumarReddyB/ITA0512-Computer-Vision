#14
from google.colab import files
import cv2
import numpy as np
from matplotlib import pyplot as plt

uploaded = files.upload()
file_name = list(uploaded.keys())[0]

img = cv2.imread(file_name)
rows, cols = img.shape[:2]


pts1 = np.float32([[56,65], [368,52], [28,387], [389,390]])
pts2 = np.float32([[0,0], [300,0], [0,300], [300,300]])


M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (300,300))

plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.title("Perspective Transformation")
plt.axis('off')
plt.show()
