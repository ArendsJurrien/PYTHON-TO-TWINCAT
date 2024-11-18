import cv2
import numpy as np

# Laad de afbeelding in grijswaarden
image = cv2.imread('big_gear flits v0.1.jpg', cv2.IMREAD_GRAYSCALE)

# Pas Gaussian blur toe om ruis te verminderen (optioneel, afhankelijk van de kwaliteit van de afbeelding)
blurred = cv2.GaussianBlur(image, (15, 15), 0)

# Gebruik Canny edge detection om randen te detecteren
edges = cv2.Canny(blurred, 50, 150)

# Zoek de contouren in de afbeelding
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Selecteer de grootste contour als de buitenste rand van het tandwiel (aangenomen dat het de grootste contour is)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Optioneel: teken de gevonden contour op de originele afbeelding voor controle
    image_contour = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_contour, [largest_contour], -1, (0, 255, 0), 2)
    
    # Zet de contour-coördinaten om in een lijst van punten
    contour_coordinates = largest_contour.reshape(-1, 2)
    
    # Print de coördinaten of stuur ze door voor verdere verwerking
    print("Contour coördinaten van de buitenste rand van het tandwiel:")
    for coord in contour_coordinates:
        print(coord)
    
    # Toon de afbeelding met de gedetecteerde contour
    scale_percent = 30  # Verkleinen tot 50% van het origineel
    width = int(image_contour.shape[1] * scale_percent / 100)
    height = int(image_contour.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image_contour, (width, height))
    cv2.imshow('Contour van tandwiel', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Geen contouren gevonden.")

# Opslaan of verdere verwerking van 'contour_coordinates' zoals het omzetten naar een pad
