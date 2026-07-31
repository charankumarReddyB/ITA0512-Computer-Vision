#34
import numpy as np
import cv2
import matplotlib.pyplot as plt
s = int(input("Enter image size: "))
img = np.ones((s, s, 3), dtype=np.uint8) * 255
center = (s//2, s//2)
radius = s//5
cv2.circle(img, center, radius, (0,255,0), 3)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Circle")
plt.axis('off')
plt.show()
