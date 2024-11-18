import cv2
import numpy as np
import matplotlib.pyplot as plt

# Open de laptopcamera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Kan de camera niet openen")
    exit()

# Neem een enkele afbeelding
ret, frame = camera.read()

if not ret:
    print("Kan geen afbeelding lezen")
    camera.release()
    exit()

# Sluit de camera
camera.release()

# Converteer de afbeelding naar grijswaarden
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Voer Canny edge detection uit
edges = cv2.Canny(gray_frame, 100, 200)

# Toon de originele afbeelding en de randen
plt.figure(figsize=(12, 6))

# Originele afbeelding
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title('Originele Afbeelding')
plt.axis('off')

# Afbeelding met randen
plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')

plt.show()
