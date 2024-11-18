import cv2
import numpy as np

# Definieer de schaal in pixels per millimeter (px_per_mm)
# Deze waarde moet bepaald worden door kalibratie
px_per_mm = 38.8  # Voorbeeld: 10 pixels per millimeter

# Open de externe camera
camera = cv2.VideoCapture(2)

if not camera.isOpened():
    print("Kan de camera niet openen. Probeer een ander cameranummer.")
    exit()

# Lees een frame van de camera
ret, frame = camera.read()
camera.release()

if not ret:
    print("Kan geen frame lezen van de camera.")
    exit()

# Converteer de afbeelding naar grijswaarden en detecteer randen
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_frame, 175, 175)

# Vind de contouren
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Selecteer de grootste contour (aanname: dit is het object)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Bereken de ingesloten cirkel om de diameter te vinden
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    diameter_px = 2 * radius
    
    # Converteer diameter naar millimeters
    diameter_mm = diameter_px / px_per_mm
    
    # Teken de contour en cirkel op de afbeelding voor controle
    contour_image = frame.copy()
    cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)  # Groene contour
    cv2.circle(contour_image, (int(x), int(y)), int(radius), (255, 0, 0), 2)  # Blauwe cirkel
    
    # Toon resultaten
    print(f"Diameter in pixels: {diameter_px}")
    print(f"Diameter in millimeters: {diameter_mm:.2f} mm")
    
    cv2.imshow('Originele Afbeelding', frame)
    cv2.imshow('Contouren en cirkel', contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Geen contouren gevonden.")
