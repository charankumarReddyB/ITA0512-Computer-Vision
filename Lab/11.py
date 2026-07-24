#11
from google.colab import files
import cv2
from matplotlib import pyplot as plt

uploaded = files.upload()
file_name = list(uploaded.keys())[0]

img = cv2.imread(file_name)

rotated = cv2.rotate(img, cv2.ROTATE_180)

plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
plt.title("180 Degree Rotation")
plt.axis('off')
plt.show()
