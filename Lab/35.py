#35
import numpy as np
import cv2
import matplotlib.pyplot as plt
text = input("Enter text: ")
img = np.ones((400,600,3),dtype=np.uint8)*255
cv2.putText(img,text,(50,200),
cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
