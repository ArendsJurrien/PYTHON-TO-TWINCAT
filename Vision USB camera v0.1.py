import cv2
import numpy as np

# Open de externe camera (meestal is 1 de index voor een externe camera, probeer 0 of 2 als dit niet werkt)
camera = cv2.VideoCapture(2)

if not camera.isOpened():
    print("Kan de camera niet openen. Probeer een ander cameranummer.")
    exit()

# Lees een frame van de camera
ret, frame = camera.read()

if not ret:
    print("Kan geen frame lezen van de camera.")
    camera.release()
    exit()

# Sluit de camera na het vastleggen van het frame
camera.release()

# Converteer de afbeelding naar grijswaarden
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Voer Canny edge detection uit om randen te vinden
edges = cv2.Canny(gray_frame, 175, 175)

# Vind de contouren in de afbeelding
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Teken de contouren op een nieuwe afbeelding
contour_image = np.zeros_like(frame)
for contour in contours:
    cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)  # Groene contouren



# Toon de originele afbeelding, randen en contouren
cv2.imshow('Originele Afbeelding', frame)
cv2.imshow('Randen', edges)
cv2.imshow('Contouren', contour_image)


# Wacht tot een toets wordt ingedrukt om de vensters te sluiten
cv2.waitKey(0)
cv2.destroyAllWindows()
