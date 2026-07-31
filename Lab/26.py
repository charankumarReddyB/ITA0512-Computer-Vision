#26
from google.colab import files
import cv2

uploaded = files.upload()
file_name = list(uploaded.keys())[0]

cap = cv2.VideoCapture(file_name)

frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

height, width = frames[0].shape[:2]

out = cv2.VideoWriter('reverse.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      20, (width,height))

for frame in reversed(frames):
    out.write(frame)

out.release()

print("Reverse video saved as reverse.mp4")
print("Reverse video saved as reverse.mp4")

from google.colab import files
files.download('reverse.mp4')
