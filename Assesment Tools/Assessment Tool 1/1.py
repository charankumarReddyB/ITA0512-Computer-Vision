import cv2
import numpy as np
image = cv2.imread("road.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blur, 50, 150)

# Detect lane lines
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                        minLineLength=50, maxLineGap=20)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1,y1), (x2,y2), (0,255,0), 3)
cv2.imshow("Lane Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()