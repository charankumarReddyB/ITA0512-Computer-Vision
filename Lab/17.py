#17
from google.colab import files
import cv2
from matplotlib import pyplot as plt

uploaded = files.upload()
file_name = list(uploaded.keys())[0]

img = cv2.imread(file_name)

# Add watermark text
watermarked = img.copy()
cv2.putText(watermarked, "WATERMARK", (50,50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

plt.imshow(cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB))
plt.title("Watermarked Image")
plt.axis('off')
plt.show()
