#39
from google.colab import files
import cv2
f = list(files.upload().keys())[0]
cap = cv2.VideoCapture(f)
frames=[]
while True:
    r,frame = cap.read()
    if not r:
        break
    frames.append(frame)
h,w = frames[0].shape[:2]
out = cv2.VideoWriter(
'reverse.mp4',
cv2.VideoWriter_fourcc(*'mp4v'),
5,(w,h))
for frame in reversed(frames):
    out.write(frame)
out.release()
print("reverse.mp4 saved")
