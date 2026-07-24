#3
from google.colab import files
import cv2
from matplotlib import pyplot as plt
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
img = cv2.imread(file_name)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blur, 50, 150)
plt.imshow(edges, cmap='gray')
plt.title("Edges")
plt.axis('off')
plt.show()
