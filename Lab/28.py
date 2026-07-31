#28
from google.colab import files
import cv2

f = list(files.upload().keys())[0]

cap = cv2.VideoCapture(f)

while True:
    r,frame = cap.read()

    if not r:
        break

    g = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    c = cv2.Canny(g,100,200)

    print("Vehicle frame detected")

cap.release()
