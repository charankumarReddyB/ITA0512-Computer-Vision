#13
from google.colab import files
import cv2
import numpy as np
from matplotlib import pyplot as plt

uploaded = files.upload()
file_name = list(uploaded.keys())[0]

img = cv2.imread(file_name)
rows, cols = img.shape[:2]


pts1 = np.float32([[50,50], [200,50], [50,200]])
pts2 = np.float32([[10,100], [200,50], [100,250]])


M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))

plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.title("Affine Transformation")
plt.axis('off')
plt.show()
