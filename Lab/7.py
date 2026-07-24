#7
import cv2
from tkinter import Tk, filedialog
Tk().withdraw()
video_path = filedialog.askopenfilename(title="Select Video File")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Video Playback", frame)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('s'):   
        cv2.waitKey(100)
    elif key == ord('f'):
        cv2.waitKey(5)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
