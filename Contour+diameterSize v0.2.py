import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lees de afbeelding in
image = cv2.imread('Small_gear zonder flits v0.1.jpg')

# Converteer de afbeelding naar grijstinten
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gebruik een Gaussiaanse blur om ruis te verminderen
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Gebruik Canny edge detection om de randen te detecteren
edges = cv2.Canny(blurred, 100, 300)

# Vind de contouren op basis van de gedetecteerde randen
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Maak een kopie van de originele afbeelding om de contouren te tekenen
image_contours = image.copy()

# Teken de contouren
cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)

# Toon de originele afbeelding en de afbeelding met contouren
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Originele afbeelding')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title('Contouren gedetecteerd')
plt.imshow(cv2.cvtColor(image_contours, cv2.COLOR_BGR2RGB))
plt.show()
