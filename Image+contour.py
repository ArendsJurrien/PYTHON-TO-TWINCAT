import cv2
import numpy as np

# Laad de tandwielafbeelding
image = cv2.imread('Small_gear flits v0.1.jpg')

# Controleer of de afbeelding correct is geladen
if image is None:
    print("Kan de afbeelding niet laden. Controleer het pad.")
    exit()

# Converteer de afbeelding naar grijswaarden
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Voer Canny edge detection uit
edges = cv2.Canny(gray_image, 100, 200)

# Vind de contouren in de afbeelding
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sla de contourcoördinaten op in een tekstbestand
with open('contour_coordinates.txt', 'w') as f:
    for contour in contours:
        # Schrijf elke coördinaat in het bestand
        for point in contour:
            f.write(f"{point[0][0]}, {point[0][1]}\n")

# Teken de contouren voor visuele verificatie
contour_image = np.zeros_like(image)
for contour in contours:
    cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)  # Groene contouren

# Toon de originele afbeelding, randen en contouren
cv2.imshow('Originele Afbeelding', image)
cv2.imshow('Randen', edges)
cv2.imshow('Contouren', contour_image)

# Wacht tot een toets wordt ingedrukt
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Contourcoördinaten zijn opgeslagen in 'contour_coordinates.txt'")
