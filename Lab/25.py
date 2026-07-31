#25
from google.colab import files
import cv2, matplotlib.pyplot as plt

f = list(files.upload().keys())[0]
img = cv2.imread(f)

g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
e = cv2.Canny(g,100,200)

c,_ = cv2.findContours(e,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for i in c:
    x,y,w,h = cv2.boundingRect(i)
    if w>50 and h>50:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
