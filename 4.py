import cv2
import numpy as np
import pyads
import time

# Open the external camera (typically 1 is the index for an external camera)
camera = cv2.VideoCapture(2)

if not camera.isOpened():
    print("Could not open the camera. Try a different camera index.")
    exit()

# Connect to the TwinCAT PLC using ADS
plc = pyads.Connection('192.168.1.10.1.1', 851)  # Replace with the correct IP address
plc.open()

# Pixel size (in microns)
pixel_size_um = 28  # Adjust to the value for your camera

# Function to convert from pixels to millimeters
def pixel_to_mm(pixel_size_um, pixel_value):
    return pixel_value * (pixel_size_um / 1000)  # Convert to millimeters

# Simulation mode flag (default is False, will be toggled later)
simulation_mode = False

# Infinite loop to process video frames
while True:
    # Read a frame from the camera
    ret, frame = camera.read()
    
    if not ret:
        print("Could not read frame from the camera.")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform Canny edge detection
    edges = cv2.Canny(gray_frame, 175, 175)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process the contours and optionally send coordinates to TwinCAT
    contour_coordinates_mm = []

    for contour in contours:
        for point in contour:
            x, y = point[0]
            
            # Convert the coordinates to millimeters
            x_mm = pixel_to_mm(pixel_size_um, x)
            y_mm = pixel_to_mm(pixel_size_um, y)
            
            # If simulation mode is on, send coordinates to TwinCAT
            if simulation_mode:
                plc.write_by_name('MAIN.GVL.xCoordinate', x_mm, pyads.PLCTYPE_REAL)
                plc.write_by_name('MAIN.GVL.yCoordinate', y_mm, pyads.PLCTYPE_REAL)
            
            contour_coordinates_mm.append([x_mm, y_mm])

    # Optional: You can visualize the results
    contour_image = np.zeros_like(frame)
    for contour in contours:
        cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)  # Green contours

    cv2.imshow('Contour Image', contour_image)

    # Check if the 's' key is pressed to toggle simulation mode
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Exit the loop if 'q' is pressed
        break
    elif key == ord('s'):  # Toggle simulation mode when 's' is pressed
        simulation_mode = not simulation_mode
        print(f"Simulation Mode: {'ON' if simulation_mode else 'OFF'}")

# Release the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

# Close the connection to TwinCAT PLC
plc.close()
