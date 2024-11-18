import cv2
import numpy as np
import time

def measure_gear_diameter(camera_index=0, pixels_per_mm=10):
    # Open de camera
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Kan de Arducam IX477 12MP camera niet openen.")
        return None

    # Wacht even zodat de camera kan opstarten
    time.sleep(2)

    # Neem een afbeelding van de camera
    ret, frame = camera.read()

    if not ret:
        print("Kan geen frame lezen van de camera.")
        camera.release()
        return None

    # Convert naar grijswaarden
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Vind de contouren
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Zoek de grootste contour (vermoedelijk de buitenrand van het tandwiel)
    largest_contour = max(contours, key=cv2.contourArea)

    # Bereken de uiterste coördinaten
    x_coordinates = largest_contour[:, 0, 0]
    min_x = np.min(x_coordinates)  # Meest linker x-coördinaat
    max_x = np.max(x_coordinates)  # Meest rechter x-coördinaat

    # Bepaal de diameter in pixels
    diameter_in_pixels = max_x - min_x

    # Bereken de diameter in millimeters
    diameter_in_mm = diameter_in_pixels / pixels_per_mm
    # Toon de resultaten
    cv2.putText(frame, f"Diameter: {diameter_in_mm:.2f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Toon de originele afbeelding en de edges
    cv2.imshow('Originele afbeelding', frame)
    cv2.imshow('Canny Edges', edges)

    # Print de resultaten naar de console
    print(f"Meest linker x-coördinaat: {min_x}")
    print(f"Meest rechter x-coördinaat: {max_x}")
    print(f"Diameter in pixels: {diameter_in_pixels}")
    print(f"Diameter in mm: {diameter_in_mm:.2f}")

    # Wacht op een toetsdruk om het venster te sluiten
    cv2.waitKey(0)

    # Sluit de camera en alle OpenCV vensters
    camera.release()
    cv2.destroyAllWindows()

# Voer de functie uit met de juiste camera-index en pixels per mm
measure_gear_diameter(camera_index=0, pixels_per_mm=10)  # Pas de camera_index en pixels_per_mm aan indien nodig
