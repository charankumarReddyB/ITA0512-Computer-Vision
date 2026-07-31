#29
from google.colab import files
import cv2, matplotlib.pyplot as plt

f = list(files.upload().keys())[0]
img = cv2.imread(f)

eye = cv2.CascadeClassifier(
cv2.data.haarcascades + 'haarcascade_eye.xml')

g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

for (x,y,w,h) in eye.detectMultiScale(g,1.1,3):
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
