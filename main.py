from image_utils import load_image, edge_detection
from skimage.filters import median
from skimage.morphology import ball, remove_small_objects, opening, disk
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


image = load_image("my_image.jpg.jpeg")


clean_image = median(image, ball(3))


edges = edge_detection(clean_image)


edge_binary = edges > 50


edge_binary = remove_small_objects(edge_binary, min_size=100)


edge_binary = opening(edge_binary, disk(1))


plt.imshow(edge_binary, cmap="gray")
plt.axis("off")
plt.show()


edge_image = Image.fromarray((edge_binary * 255).astype(np.uint8))
edge_image.save("my_edges.png")
