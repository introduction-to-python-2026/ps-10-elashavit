from image_utils import load_image, edge_detection
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Step 1: load original image
image = load_image("my_image.jpg")

# Step 2: suppress noise
clean_image = median(image, ball(3))

# Step 3: edge detection
edges = edge_detection(clean_image)

# Step 4: convert to binary
edge_binary = edges > 50

# Step 5: display result
plt.imshow(edge_binary, cmap="gray")
plt.axis("off")
plt.show()

# Step 6: save edge image
edge_image = Image.fromarray((edge_binary * 255).astype(np.uint8))
edge_image.save("my_edges.png")

