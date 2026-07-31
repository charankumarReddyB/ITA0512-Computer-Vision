#21
from google.colab import files
import cv2
import numpy as np
from matplotlib import pyplot as plt

uploaded = files.upload()
file_name = list(uploaded.keys())[0]

img = cv2.imread(file_name)

kernel = np.ones((5,5), np.uint8)

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

plt.imshow(cv2.cvtColor(opening, cv2.COLOR_BGR2RGB))
plt.title("Opening Operation")
plt.axis('off')
plt.show()
