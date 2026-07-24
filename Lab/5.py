#5
from google.colab import files
import cv2
from matplotlib import pyplot as plt
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
def plot_histogram(path):
    img = cv2.imread(path)
    if img is None:
        print("Image not loaded")
        return
    colors = ('b','g','r')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0,256])
        plt.plot(hist, color=col)
    plt.title("Color Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Pixels")
    plt.show()
plot_histogram(file_name)
