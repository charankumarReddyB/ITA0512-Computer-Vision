#27
from google.colab import files
import cv2
import matplotlib.pyplot as plt

f = list(files.upload().keys())[0]
img = cv2.imread(f)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

faces = face.detectMultiScale(gray, 1.05, 2)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

print("Faces detected:", len(faces))

plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
