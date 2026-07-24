#15
from google.colab import files
import cv2
import numpy as np
from matplotlib import pyplot as plt

uploaded = files.upload()
file_name = list(uploaded.keys())[0]

img = cv2.imread(file_name)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)


dst = cv2.cornerHarris(gray, 2, 3, 0.04)


img[dst > 0.01 * dst.max()] = [0, 0, 255]

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Harris Corners")
plt.axis('off')
plt.show()
