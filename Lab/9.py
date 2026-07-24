#9
from google.colab import files
import cv2
from matplotlib import pyplot as plt

uploaded = files.upload()
file_name = list(uploaded.keys())[0]

img = cv2.imread(file_name)


small = cv2.resize(img, (200,200))


big = cv2.resize(img, (800,800))

plt.imshow(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
plt.title("Scaled Image")
plt.axis('off')
plt.show()
