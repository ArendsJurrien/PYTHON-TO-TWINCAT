import cv2                      # OpenCV-bibliotheek voor beeldverwerking 
import pyrealsense2 as rs       # Intel RealSense-bibliotheek voor camera-invoer
import numpy as np              # NumPy voor numerieke berekeningen
import time                     # Voor tijdsvertragingen
import csv                      # Voor het schrijven van CSV-bestanden
import pyads                    # Voor communicatie met een TwinCAT PLC

# Functie voor het schrijven van de coördinaten naar TwinCAT
def send_coordinates_to_twincat(x_coords, y_coords, plc_address="39.231.85.117.1.1", port=851): 
    try:
        plc = pyads.Connection(plc_address, port)
        plc.open()
        if plc.is_open:
            print(f"Connected to PLC at {plc_address} on port {port}")
            for i, (x, angle) in enumerate(zip(x_coords, y_coords)):
                if i < 9999:
                    plc.write_by_name(f'Main.x_coords[{i + 1}]', x, pyads.PLCTYPE_REAL)
                    plc.write_by_name(f'Main.y_coords[{i + 1}]', angle, pyads.PLCTYPE_REAL)
            print("Coördinaten succesvol verzonden naar PLC.")
        else:
            print("Failed to open connection to PLC.")
        plc.close()
    except Exception as e:
        print(f"Error: {e}")

# Initialiseer Intel RealSense D435 camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Parameters
lower_threshold = 25
upper_threshold = 150
pixels_per_mm_at_reference_distance = 2.45
reference_distance_m = 0.25
min_diameter_mm = 40
max_diameter_mm = 180
csv_filename = "contour_coordinates_mm_&_degrees.csv"

try:
    while True:
        time.sleep(2)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("Geen frames ontvangen. Probeer opnieuw.")
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        edges = cv2.Canny(blurred, lower_threshold, upper_threshold)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)

            vertical_x = int(x)
            start_point = min(largest_contour, key=lambda p: abs(p[0][0] - vertical_x))
            start_index = np.where((largest_contour == start_point).all(axis=2))[0][0]
            reordered_contour = np.concatenate((largest_contour[start_index:], largest_contour[:start_index]))

            circle_area = np.pi * (radius ** 2)
            contour_area = cv2.contourArea(largest_contour)
            if 0.7 < contour_area / circle_area < 1.0:
                depth_at_center = depth_frame.get_distance(int(x), int(y))

                if depth_at_center > 0:
                    pixels_per_mm = pixels_per_mm_at_reference_distance * (reference_distance_m / depth_at_center)
                    diameter_mm = (2 * radius) / pixels_per_mm

                    if min_diameter_mm <= diameter_mm <= max_diameter_mm:
                        center_x_mm = x / pixels_per_mm
                        center_y_mm = y / pixels_per_mm

                        cv2.drawContours(color_image, [reordered_contour], -1, (0, 255, 0), 2)
                        cv2.circle(color_image, (int(x), int(y)), int(radius), (255, 0, 0), 2)
                        cv2.line(color_image, (int(x), 0), (int(x), color_image.shape[0]), (255, 0, 0), 2)
                        cv2.putText(color_image, "X-axis", (int(x) + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.line(color_image, (0, int(y)), (color_image.shape[1], int(y)), (0, 255, 0), 2)
                        cv2.putText(color_image, "Y-axis", (color_image.shape[1] - 100, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        x_coords_mm = []
                        y_coords_deg = []
                        for point in reordered_contour:
                            pixel_x, pixel_y = point[0]
                            mm_x = (pixel_x - x) / pixels_per_mm
                            mm_y = (pixel_y - y) / pixels_per_mm

                            distance = np.sqrt(mm_x**2 + mm_y**2)
                            angle_rad = np.arctan2(mm_y, mm_x)
                            angle_deg = np.degrees(angle_rad)
                            # Verplaats de 0 graden naar boven
                            angle_deg = (angle_deg + 90) % 360

                            x_coords_mm.append(distance)
                            y_coords_deg.append(angle_deg)

                        with open(csv_filename, mode="w", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow(["Distance (mm)", "Angle (degrees)"])
                            for x, y in zip(x_coords_mm, y_coords_deg):
                                writer.writerow([x, y])
                        
                        send_coordinates_to_twincat(x_coords_mm, y_coords_deg)

                        # Markeer het eerste punt en pas de hoek aan voor de visualisatie
                        first_point_x, first_point_y = start_point[0]
                        cv2.circle(color_image, (first_point_x, first_point_y), 3, (0, 0, 255), -1)
                        cv2.putText(color_image, "First Point", (first_point_x + 10, first_point_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                        # Visualiseer het eerste punt van x_coords_mm in het beeld
                        if x_coords_mm:  # Controleer of de lijst niet leeg is
                            first_x = x_coords_mm[0]
                            first_angle = y_coords_deg[0]

                            # Bepaal de pixelcoördinaten van het eerste punt (om te tekenen)
                            first_point_x = int(x + first_x * pixels_per_mm)
                            first_point_y = int(y)

                            # Toon de coördinaten van het eerste punt op het beeld
                            cv2.putText(color_image, f"First X: {first_x:.2f} mm, Angle: {first_angle:.2f} degrees",
                                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        # Pas de hoek van het eerste punt aan zodat 0 graden naar boven wijst
                        angle_deg_first_point = (np.degrees(np.arctan2(first_point_y - y, first_point_x - x)) + 90) % 360
                        print(f"Adjusted angle for first point: {angle_deg_first_point:.2f} degrees")

                        cv2.putText(color_image, f"Diameter: {diameter_mm:.2f} mm",
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                        cv2.imshow("Tandwiel Resultaat", color_image)
                        cv2.imwrite("resultaat_tandwiel.png", color_image)
                        cv2.waitKey(0)
                        break
                    else:
                        print(f"Contour genegeerd: diameter ({diameter_mm:.2f} mm) ligt niet tussen {min_diameter_mm} mm en {max_diameter_mm} mm.")
                else:
                    print("Geen object gevonden.")
        else:
            print("Geen contouren gevonden. Controleer het beeld en probeer opnieuw.")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
