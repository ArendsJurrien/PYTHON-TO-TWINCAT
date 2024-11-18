import cv2
import numpy as np

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

# Vind de contouren in de afbeelding
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Maak een nieuwe afbeelding om de contouren op te tekenen
contour_image = np.zeros_like(frame)

# Teken de contouren
for contour in contours:
    cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)  # Groene contouren

# Toon de originele afbeelding, randen en contouren
cv2.imshow('Originele Afbeelding', frame)
cv2.imshow('Randen', edges)
cv2.imshow('Contouren', contour_image)

# Wacht tot een toets wordt ingedrukt
cv2.waitKey(0)
cv2.destroyAllWindows()
