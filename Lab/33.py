#33
import numpy as np
import cv2
import matplotlib.pyplot as plt
s = int(input("Enter image size: "))
img = np.ones((s, s, 3), dtype=np.uint8) * 255
start_point = (s//4, s//4)
end_point = (3*s//4, s//4 + s//3)
cv2.rectangle(img, start_point, end_point, (255,0,0), 3)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Rectangle")
plt.axis('off')
plt.show()
