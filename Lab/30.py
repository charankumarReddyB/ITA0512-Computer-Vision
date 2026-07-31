#30
from google.colab import files
import cv2
import matplotlib.pyplot as plt

f = list(files.upload().keys())[0]

img = cv2.imread(f)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

smile = cv2.CascadeClassifier(
cv2.data.haarcascades + 'haarcascade_smile.xml')

smiles = smile.detectMultiScale(
gray,
scaleFactor=1.3,
minNeighbors=5
)

for (x,y,w,h) in smiles:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

print("Smiles detected:",len(smiles))

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
