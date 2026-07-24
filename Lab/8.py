#8
from google.colab import files
import cv2
from matplotlib import pyplot as plt
import numpy as np

uploaded = files.upload()
file_name = list(uploaded.keys())[0]

img = cv2.imread(file_name)

kernel = np.ones((5,5), np.uint8)

dilated = cv2.dilate(img, kernel, iterations=1)

plt.imshow(cv2.cvtColor(dilated, cv2.COLOR_BGR2RGB))
plt.title("Dilated Image")
plt.axis('off')
plt.show()
