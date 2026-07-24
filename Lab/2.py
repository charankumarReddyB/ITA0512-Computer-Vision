#2
from google.colab import files
import cv2
from matplotlib import pyplot as plt
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
img = cv2.imread(file_name)
blur = cv2.GaussianBlur(img, (9, 9), 0)
blur_rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
plt.imshow(blur_rgb)
plt.title("Blurred Image")
plt.axis('off')
plt.show()
