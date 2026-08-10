import cv2
import time
from pathlib import Path

script_dir = Path(__file__).resolve().parent
image_path = script_dir / "image.jpg"

if not image_path.is_file():
    raise FileNotFoundError(f"Image file not found: {image_path}")

image = cv2.imread(str(image_path))

if image is None:
    raise ValueError(f"OpenCV could not read the image file: {image_path}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

settings = [250, 500, 1000, 2000]

height, width = gray.shape
area = height * width

print("| Maximum Features | Detected Features | Feature Density | Detection Time |")
print("| ---------------: | ----------------: | --------------: | -------------: |")

for n in settings:

    orb = cv2.ORB_create(nfeatures=n)

    start = time.perf_counter()

    keypoints, descriptors = orb.detectAndCompute(
        gray, None
    )

    end = time.perf_counter()

    count = len(keypoints)

    density = count / (area / 1000000)

    time_ms = (end - start) * 1000

    print(
        f"| {n:16d} | {count:17d} | {round(density, 2):14.2f} | {time_ms:13.3f} ms |"
    )