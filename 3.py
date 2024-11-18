import cv2
import numpy as np
import csv

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

# Pixel grootte (in microns)
pixel_size_um = 28  # Stel de pixelgrootte in microns in, dit kun je aanpassen naar de waarde voor jouw camera

# Functie om van pixels naar millimeters om te rekenen
def pixel_to_mm(pixel_size_um, pixel_value):
    return pixel_value * (pixel_size_um / 1000)  # Omrekenen naar millimeters

# Sla de coördinaten op in een lijst
contour_coordinates_mm = []

# Maak een lege afbeelding voor het tekenen van contouren
contour_image = np.zeros_like(frame)

for contour in contours:
    for point in contour:
        # Verkrijg de coördinaten van elk punt van de contour
        x, y = point[0]
        
        # Zet de coördinaten om naar millimeters
        x_mm = pixel_to_mm(pixel_size_um, x)
        y_mm = pixel_to_mm(pixel_size_um, y)
        
        # Voeg de coördinaten toe aan de lijst
        contour_coordinates_mm.append([x_mm, y_mm])

        # Teken de contouren op de contour_image
        cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)  # Groene contouren

# Sla de coördinaten op in een CSV-bestand
with open('contour_coordinates.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['X (mm)', 'Y (mm)'])  # Schrijf header
    writer.writerows(contour_coordinates_mm)  # Schrijf de coördinaten

print("Coördinaten zijn opgeslagen in 'contour_coordinates.csv'")

# Toon de originele afbeelding, randen en contouren
cv2.imshow('Originele Afbeelding', frame)
cv2.imshow('Randen', edges)
cv2.imshow('Contouren', contour_image)

# Wacht tot een toets wordt ingedrukt om de vensters te sluiten
cv2.waitKey(0)
cv2.destroyAllWindows()
