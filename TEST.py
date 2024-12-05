import cv2
import pyrealsense2 as rs
import numpy as np
import time
import csv
import pyads

# Functie voor het schrijven van de coördinaten naar TwinCAT
def send_coordinates_to_twincat(x_coords, y_coords, plc_address="39.231.85.117.1.1", port=851):
    # Verzendt de coördinaten (x en hoek) naar TwinCAT via ADS.
    try:
        plc = pyads.Connection(plc_address, port)  # Maak verbinding met de PLC
        plc.open()  # Open de verbinding

        if plc.is_open:
            print(f"Connected to PLC at {plc_address} on port {port}")
            for i, (x, angle) in enumerate(zip(x_coords, y_coords)):
                if i < 9999:  # Beperk tot maximaal 9999 coördinaten
                    plc.write_by_name(f'Main.x_coords[{i + 1}]', x, pyads.PLCTYPE_REAL)  # Schrijf de x-waarde naar PLC
                    plc.write_by_name(f'Main.y_coords[{i + 1}]', angle, pyads.PLCTYPE_REAL)  # Schrijf de hoek naar PLC
            print("Coördinaten succesvol verzonden naar PLC.")
        else:
            print("Failed to open connection to PLC.")
        plc.close()  # Sluit de verbinding met de PLC

    except Exception as e:
        print(f"Error: {e}")  # Feedback foutmelding

# Initialiseer Intel RealSense D435 camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Schakel kleurstream in
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Schakel dieptestream in
pipeline.start(config)  # Start de camera-stream

# Parameters
lower_threshold = 25  # Ondergrens voor Canny Edge-detectie
upper_threshold = 150  # Bovengrens voor Canny Edge-detectie
pixels_per_mm_at_reference_distance = 2.45  # Pixels per mm op referentieafstand
reference_distance_m = 0.25  # Referentieafstand in meters
min_diameter_mm = 40  # Minimale toegestane diameter van een tandwiel
max_diameter_mm = 180  # Maximale toegestane diameter van een tandwiel
csv_filename = "contour_coordinates_mm_&_degrees.csv"  # Naam van het CSV-bestand

try:
    while True:
        time.sleep(2)  # Wacht 2 seconden ivm licht
        frames = pipeline.wait_for_frames()  # Wacht op een nieuwe frame-set
        color_frame = frames.get_color_frame()  # Haal de kleurinformatie op
        depth_frame = frames.get_depth_frame()  # Haal de diepte-informatie op

        if not color_frame or not depth_frame:  # Controleer of frames geldig zijn
            print("Geen frames ontvangen. Probeer opnieuw.")
            continue  # Ga terug naar het begin van de lus

        color_image = np.asanyarray(color_frame.get_data())  # Zet kleurframe om naar een NumPy-array
        depth_image = np.asanyarray(depth_frame.get_data())  # Zet diepteframe om naar een NumPy-array

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)  # Converteer naar grijswaarden
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)  # Verwijder ruis met Gaussian Blur
        edges = cv2.Canny(blurred, lower_threshold, upper_threshold)  # Detecteer randen met Canny

        # Vind contouren
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Vind externe contouren

        if contours:  # Controleer of er contouren zijn gevonden
            largest_contour = max(contours, key=cv2.contourArea)  # Selecteer het grootste contour
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)  # Bepaal de omschrijvende cirkel

            # Vind het eerste punt op de verticale x-as (hoogste punt op de positieve y-as)
            start_point = max(largest_contour, key=lambda p: p[0][1])  # Kies het punt met de hoogste y-coördinaat
            start_index = np.where((largest_contour == start_point).all(axis=2))[0][0]  # Vind index van het punt

            # Herordenen van het contour zodat het start bij dit punt
            reordered_contour = np.concatenate((largest_contour[start_index:], largest_contour[:start_index]))

            # Diameterberekening zoals eerder
            circle_area = np.pi * (radius ** 2)  # Bereken de oppervlakte van de cirkel
            contour_area = cv2.contourArea(largest_contour)  # Bereken de oppervlakte van het contour
            if 0.7 < contour_area / circle_area < 1.0:  # Controleer de verhouding contour/cirkel
                depth_at_center = depth_frame.get_distance(int(x), int(y))  # Meet de diepte bij het middelpunt

                if depth_at_center > 0:  # Controleer of de diepte geldig is
                    pixels_per_mm = pixels_per_mm_at_reference_distance * (reference_distance_m / depth_at_center)  # Schaalpixels
                    diameter_mm = (2 * radius) / pixels_per_mm  # Bereken diameter in mm

                    if min_diameter_mm <= diameter_mm <= max_diameter_mm:  # Controleer diametergrenzen
                        center_x_mm = x / pixels_per_mm  # Zet x-coördinaat om naar mm
                        center_y_mm = y / pixels_per_mm  # Zet y-coördinaat om naar mm

                        # Visualisatie
                        cv2.drawContours(color_image, [reordered_contour], -1, (0, 255, 0), 2)  # Teken herordend contour
                        cv2.circle(color_image, (int(x), int(y)), int(radius), (255, 0, 0), 2)  # Teken omschrijvende cirkel

                        # Teken de verticale (x-as) en horizontale (y-as)
                        cv2.line(color_image, (int(x), 0), (int(x), color_image.shape[0]), (255, 0, 0), 2)  # x-as
                        cv2.putText(color_image, "X-axis", (int(x) + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.line(color_image, (0, int(y)), (color_image.shape[1], int(y)), (0, 255, 0), 2)  # y-as
                        cv2.putText(color_image, "Y-axis", (color_image.shape[1] - 100, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        # Bepaal de afstand en hoeken ten opzichte van het eerste punt
                        x_coords_mm = []
                        y_coords_deg = []
                        for point in reordered_contour:
                            pixel_x, pixel_y = point[0]
                            mm_x = (pixel_x - x) / pixels_per_mm
                            mm_y = (pixel_y - y) / pixels_per_mm

                            distance = np.sqrt(mm_x**2 + mm_y**2)
                            angle_rad = np.arctan2(mm_y, mm_x)
                            angle_deg = np.degrees(angle_rad)
                            # Om de referentiehoek naar boven (0 graden naar boven) te verplaatsen
                            angle_deg = (angle_deg + 90) % 360  # Verplaats de 0 graden naar boven

                            x_coords_mm.append(distance)
                            y_coords_deg.append(angle_deg)

                        with open(csv_filename, mode="w", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow(["Distance (mm)", "Angle (degrees)"])
                            for x, y in zip(x_coords_mm, y_coords_deg):
                                writer.writerow([x, y])

                        # Markeer het eerste punt
                        first_point_x, first_point_y = start_point[0]
                        cv2.circle(color_image, (first_point_x, first_point_y), 3, (0, 0, 255), -1)  # Rood stipje
                        cv2.putText(color_image, "First Point", (first_point_x + 10, first_point_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                        cv2.putText(color_image, f"Diameter: {diameter_mm:.2f} mm",
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                        cv2.imshow("Tandwiel Resultaat", color_image)
                        cv2.imwrite("resultaat_tandwiel.png", color_image)
                        cv2.waitKey(1)

                        send_coordinates_to_twincat(x_coords_mm, y_coords_deg)

        else:
            print("Geen contouren gevonden.")

except KeyboardInterrupt:
    print("Programma beëindigd door gebruiker.")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
