#4
from google.colab import files
import cv2
from matplotlib import pyplot as plt
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
img = cv2.imread(file_name)
if img is None:
    print("Image not loaded")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    plt.imshow(equalized, cmap='gray')
    plt.title("Equalized")
    plt.axis('off')
    plt.show()
