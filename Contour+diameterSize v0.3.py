import cv2
import numpy as np

# Stap 1: Lees de afbeelding
image = cv2.imread('Small_gear zonder flits v0.1.jpg')  # Vervang met het pad naar je afbeelding

# Stap 2: Conversie naar grijswaarden
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Stap 3: Toepassen van een Gaussiaanse blur
blurred = cv2.GaussianBlur(gray, (15, 15), 0)

# Stap 4: Randdetectie met Canny
edges = cv2.Canny(blurred, 50, 75)

# Stap 5: Zoek naar contouren
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Stap 6: Teken de contouren op de originele afbeelding
output = image.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)  # Groene contouren

# Stap 7: Toon de resultaten
cv2.imshow('Original Image', image)
cv2.imshow('Edges', edges)
cv2.imshow('Contours', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
