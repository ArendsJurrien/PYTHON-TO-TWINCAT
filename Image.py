import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('Big_gear zonder flits v0.2.jpg', cv2.IMREAD_GRAYSCALE)

# Apply GaussianBlur to reduce noise
blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

# Apply Canny edge detection
edges = cv2.Canny(blurred_image, 50, 200)

# Display the result
plt.figure(figsize=(8, 6))
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')
plt.show()

