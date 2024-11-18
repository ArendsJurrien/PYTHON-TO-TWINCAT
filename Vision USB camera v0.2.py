import cv2
import numpy as np

# Open de externe camera (meestal 1, probeer 0 of 2 indien nodig)
camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Kan de camera niet openen. Probeer een ander cameranummer.")
    exit()

while True:
    # Lees een frame van de camera
    ret, frame = camera.read()

    if not ret:
        print("Kan geen frame lezen van de camera.")
        break

    # Converteer de afbeelding naar grijswaarden
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pas een Gaussiaans filter toe om ruis te verminderen
    blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Voer Canny edge detection uit om randen te vinden
    edges = cv2.Canny(blurred, 50, 150)

    # Vind alleen de externe contouren
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Maak een nieuwe afbeelding om de contouren op te tekenen
    contour_image = np.zeros_like(frame)

    # Filter contouren op basis van grootte (we zoeken grote contouren zoals het tandwiel)
    min_contour_area = 1000  # Minimale oppervlakte voor het tandwiel
    max_contour_area = 50000  # Maximale oppervlakte om ruis te voorkomen

    # Contouren verwerken
    for contour in contours:
        area = cv2.contourArea(contour)

        # Alleen contouren binnen een bepaald gebied behouden (voor het tandwiel)
        if min_contour_area < area < max_contour_area:
            # Teken de contour van het tandwiel
            cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)

            # Bereken het aantal tanden door hoekpunten van de contour te approximeren
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Teken de approximated contour (waarschijnlijk de tanden)
            cv2.drawContours(contour_image, [approx], -1, (255, 0, 0), 2)

            # Toon het aantal tanden van het tandwiel
            num_teeth = len(approx)  # Aantal hoeken in de approximatie = aantal tanden
            print(f"Aantal tanden: {num_teeth}")
            cv2.putText(contour_image, f"Tanden: {num_teeth}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Toon de originele afbeelding, randen en contouren
    cv2.imshow('Originele Afbeelding', frame)
    cv2.imshow('Randen', edges)
    cv2.imshow('Contouren', contour_image)

    # Stop de loop als de 'q' toets wordt ingedrukt
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Sluit de camera en vernietig alle vensters
camera.release()
cv2.destroyAllWindows()
