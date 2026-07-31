#38
from google.colab import files
import cv2
f = list(files.upload().keys())[0]
img = cv2.imread(f)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
face = cv2.CascadeClassifier(
cv2.data.haarcascades +
'haarcascade_frontalface_default.xml')
faces = face.detectMultiScale(gray,1.05,3)
print("Faces:",len(faces))
