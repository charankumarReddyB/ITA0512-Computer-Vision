#23
from google.colab import files
import cv2
import numpy as np
from matplotlib import pyplot as plt

uploaded = files.upload()
file_name = list(uploaded.keys())[0]

img = cv2.imread(file_name)

kernel = np.ones((9,9), np.uint8)

tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

plt.imshow(cv2.cvtColor(tophat, cv2.COLOR_BGR2RGB))
plt.title("Top Hat")
plt.axis('off')
plt.show()
