#18
from google.colab import files
import cv2
from matplotlib import pyplot as plt

uploaded = files.upload()
file_name = list(uploaded.keys())[0]

img = cv2.imread(file_name)

# Crop ROI
roi = img[50:200, 50:200]

# Paste ROI to another location
img[250:400, 250:400] = roi

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("ROI Copy-Paste")
plt.axis('off')
plt.show()
