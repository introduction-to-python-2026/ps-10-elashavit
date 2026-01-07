from image_utils import load_image, edge_detection
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


image = load_image("my_image.jpg")


clean_image = median(image, ball(3))

edges = edge_detection(clean_image)


edge_binary = edges > 50


plt.imshow(edge_binary, cmap="gray")
plt.axis("off")
plt.show()


edge_image = Image.fromarray((edge_binary * 255).astype(np.uint8))
edge_image.save("my_edges.png")

