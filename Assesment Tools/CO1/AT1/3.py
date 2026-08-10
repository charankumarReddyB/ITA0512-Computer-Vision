import cv2
image = cv2.imread("image1.jpg")
low = cv2.resize(image, (80, 80),
                 interpolation=cv2.INTER_NEAREST)

jagged = cv2.resize(low,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
cv2.imshow("Original Image", image)
cv2.imshow("Low Sampling Effect", jagged)
cv2.waitKey(0)
cv2.destroyAllWindows()