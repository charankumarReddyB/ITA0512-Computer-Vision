#40
!apt install tesseract-ocr -y
!pip install pytesseract
from google.colab import files
import cv2
import pytesseract
f = list(files.upload().keys())[0]
cap = cv2.VideoCapture(f)
r, frame = cap.read()
if r:
    text = pytesseract.image_to_string(frame)
    print(text)
cap.release()
