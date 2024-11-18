import cv2
import numpy as np
import pyads
import csv

# Functie om X- en Y-coördinaten naar TwinCAT te verzenden
def send_coordinates_to_twincat(x_coords, y_coords, plc_address="39.231.85.117.1.1", port=851):
    try:
        plc = pyads.Connection(plc_address, port)
        plc.open()
        
        if len(x_coords) != len(y_coords):
            print("Aantal X- en Y-coördinaten komt niet overeen.")
            return
        
        plc.write_by_name("M.x_coordinate_array", x_coords, pyads.PLCTYPE_INT * len(x_coords))
        plc.write_by_name("MAIN.y_coordinate_array", y_coords, pyads.PLCTYPE_INT * len(y_coords))
        
        print("Coördinaten succesvol verzonden!")
    except Exception as e:
        print(f"Fout bij verzenden van coördinaten: {e}")
    finally:
        plc.close()

# Functie om coördinaten op te slaan in een CSV-bestand
def save_coordinates_to_csv(coordinates, filename="coordinates.csv"):
    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['X', 'Y'])
            writer.writerows(coordinates)
        print(f"Coördinaten opgeslagen in {filename}")
    except Exception as e:
        print(f"Fout bij opslaan in CSV: {e}")

# Open de camera
camera = cv2.VideoCapture(1)  # Vervang 0 door het juiste cameranummer indien nodig

if not camera.isOpened():
    print("Camera kon niet worden geopend.")
    exit()

while True:
    # Lees een frame van de camera
    ret, frame = camera.read()
    if not ret:
        print("Kon geen frame lezen van de camera.")
        break

    # Verklein de afbeelding (optioneel)
    scale_percent = 50  # Schaal naar 50% van originele grootte
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # Grijswaardenconversie en Gaussian Blur
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Thresholding
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    # Zoek contouren
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Neem de grootste contour aan als belangrijkste object
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
      