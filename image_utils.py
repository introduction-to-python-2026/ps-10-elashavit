from PIL import Image
import numpy as np
from scipy.ndimage import convolve


def load_image(path):
    ` img = Image.open(path).convert("RGB")
    return np.array(img)

def edge_detection(image):
    gray = image.mean(axis=2)

    kernelY = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])

    kernelX = np.array([
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]
    ])

    edgeX = convolve2d(gray, kernelX, mode="same", boundary="fill", fillvalue=0)
    edgeY = convolve2d(gray, kernelY, mode="same", boundary="fill", fillvalue=0)

    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG
