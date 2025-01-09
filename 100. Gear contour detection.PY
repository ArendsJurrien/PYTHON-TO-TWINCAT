## MINOR INDUSTRIAL AUTOMATION
## POST PROCESSING SYSTEM
## DESCRIPTION: CREATING, PROCESSING AND SENDING COORDINATES FROM GEAR FOR DEBURRING PROCESS
## DATE: 09-01-2025
## VERSION 2.0

## Libraries
import cv2                      # OpenCV library for image and video processing
import pyrealsense2 as rs       # Accessing depth and RGB data from the camera
import numpy as np              # For numerical operations
import time                     # Time module 
import csv                      # Reading and writing data to CSV files
import pyads                    # Library for communication with the TwinCAT PLC

## Parameters 
lower_threshold = 25            # Lower threshold for edge detection (Canny edge)
upper_threshold = 150           # Upper threshold for edge detection (Canny edge)
pixels_per_mm_at_reference_distance = 2.45  # Number of pixels per millimeter at a specific reference distance
reference_distance_m = 0.25     # Reference distance in meters
min_diameter_mm = 40            # minimum size of the gear in mm
max_diameter_mm = 180           # maximum size of the gear in mm
csv_filename = "contour_coordinates_mm_&_degrees.csv"  # Name of CSV file with contour coordinates

## Function to send the coordinates to TwinCAT
def send_coordinates_to_twincat(x_coords, y_coords, plc_address="39.231.85.117.1.1", port=851):
    try:
        plc = pyads.Connection(plc_address, port)                               # Creating a PLC connection
        plc.open()                                                              # Open the connection to the PLC
        if plc.is_open:                                                         # Check if the connection is successfully opened
            print(f"Connected to PLC at {plc_address} on port {port}")
            for i, (x, angle) in enumerate(zip(x_coords, y_coords)):            # Iterate through X and Y coordinates.
                if i < 9999:  
                    plc.write_by_name(f'Main.x_coords[{i + 1}]', x, pyads.PLCTYPE_REAL)         # Send X coordinate
                    plc.write_by_name(f'Main.y_coords[{i + 1}]', angle, pyads.PLCTYPE_REAL)     # Send angle
            print("Coordinates successfully sent to PLC.")
        else:
            print("Failed to open connection to PLC.")
        plc.close()                                                             # Close the connection to the PLC
    except Exception as e:
        print(f"Error: {e}")    

## Function to check the start pulse from TwinCAT
def check_start_pulse(plc, variable_name="Main.startprocess"):
    try:
        if plc.is_open:                                                         # Check if the PLC connection is open
            return plc.read_by_name(variable_name, pyads.PLCTYPE_BOOL)          # Read the start signal
        else:
            print("PLC connection is not open.")
            return False
    except Exception as e:
        print(f"Error while reading the start pulse: {e}")
        return False

## Function to reset the start pulse in TwinCAT
def reset_start_pulse(plc, variable_name="Main.startprocess"):
    try:
        if plc.is_open:                                                         # Check if the PLC connection is open
            plc.write_by_name(variable_name, False, pyads.PLCTYPE_BOOL)         # Set to the start signal to False
        else:
            print("PLC connection is not open.")
    except Exception as e:
        print(f"Error while resetting the start pulse: {e}")

## Function to send a status message to TwinCAT
def send_status_to_twincat(text, plc, variable_name="Main.status_message"):
    try:
        if plc and plc.is_open:                                                 # Check if the PLC connection is open
            plc.write_by_name(variable_name, text, pyads.PLCTYPE_STRING)        # Write the status
            print(f"Message '{text}' successfully sent to TwinCAT.")
        else:
            print("Unable to connect to the PLC.")
    except Exception as e:
        print(f"Error while sending: {e}")

## Function to send the next move to TwinCAT
def send_next_move_to_twincat(text, plc, variable_name="Main.next_move"):
    try:
        if plc and plc.is_open:                                                 # Check if the PLC connection is open
            plc.write_by_name(variable_name, text, pyads.PLCTYPE_STRING)        # Write the next move instruction
            print(f"Message '{text}' successfully sent to TwinCAT.")
        else:
            print("Unable to connect to the PLC.")
    except Exception as e:
        print(f"Error while sending: {e}")

## Function to set the restart variable to True in TwinCAT
def set_restart_variable_true(plc, variable_name="Main.restart", value=True):
    try:
        if plc.is_open:                                                         # Check if the PLC connection is open
            plc.write_by_name(variable_name, value, pyads.PLCTYPE_BOOL)         # Set restart to True
        else:
            print("PLC connection is not open.")
    except Exception as e:
        print(f"Error setting restart variable: {e}")

## Function to set the restart variable to False in TwinCAT
def set_restart_variable_false(plc, variable_name="Main.restart", value=False):
    try:
        if plc.is_open:                                                         # Check if the PLC connection is open
            plc.write_by_name(variable_name, value, pyads.PLCTYPE_BOOL)         # Set restart to False
        else:
            print("PLC connection is not open.")
    except Exception as e:
        print(f"Error setting restart variable: {e}")

## Function to set process completed to True in TwinCAT
def set_process_completed_true(plc, variable_name="Main.coordinatesreceived", value=True):
    try:
        if plc.is_open:                                                         # Check if the PLC connection is open
            plc.write_by_name(variable_name, value, pyads.PLCTYPE_BOOL)         # Set restart to True
        else:
            print("PLC connection is not open.")
    except Exception as e:
        print(f"Error setting restart variable: {e}")



### Main function of the programm
def main():
    plc_address = "39.231.85.117.1.1"               # Set PLC address
    port = 851                                      # Communication port
    plc = pyads.Connection(plc_address, port)       # Create PLC connection
    plc.open()                                      # Open the PLC connection

    pipeline = rs.pipeline()                                                    # Create a pipeline. Pipeline = Managing data of camera
    config = rs.config()                                                        # Create a configuration. (specify data of camera)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)         # Enable color stream at 640x480 resolution
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)          # Enable depth stream at 640x480 resolution
    pipeline.start(config)                                                      # Start the RealSense pipeline
    set_restart_variable_false(plc, variable_name="Main.restart", value=False)  # Reset 'restart' in the PLC

    try:
        processing = False 
        while True:
            if not processing:
                # Wait for the start pulse
                send_next_move_to_twincat("Waiting for start pulse...", plc)    # Send message to TwinCAT to wait for the start signal
                while not check_start_pulse(plc):                               # Loop until the start signal is detected
                    time.sleep(0.1)                                             # Check for the start every (0.1s)

                # Start run mode
                send_status_to_twincat("Searching for gear", plc)  
                send_next_move_to_twincat("Idle", plc)  
                print("Run mode started. Processing image...")  
                processing = True                                             
                reset_start_pulse(plc)  

            ## Main processing loop
            timeout_seconds = 10                                                # Timeout time for finding the gear
            start_time = time.time()                                            # Record the start time 

            while processing:
                elapsed_time = time.time() - start_time                         # Calculate the elapsed time since the process started

                # Check for a new start pulse during processing
                if check_start_pulse(plc):                                      # If a start pulse is detected:
                    send_status_to_twincat("Restarting process", plc)  
                    print("Start pulse received during processing. Restarting...") 
                    processing = False                                          # Stop the current processing loop
                    reset_start_pulse(plc)                                      # Reset the start pulse 
                    break                                                       # Exit the current processing loop and restart the main loop

                # Check for timeout
                if elapsed_time > timeout_seconds:  
                    send_status_to_twincat("Timeout occurred. No gear found.", plc)  
                    processing = False                                          # Stop the current processing loop
                    break                                                       # Exit the current processing loop and restart the main loop

                # Read frames from the camera
                frames = pipeline.wait_for_frames()                             # Wait for new frames from the Camera pipeline
                color_frame = frames.get_color_frame()                          # Extract the color frame 
                depth_frame = frames.get_depth_frame()                          # Extract the depth frame

                if not color_frame or not depth_frame:                          # Check if valid frames were received
                    print("No frames received. Retrying...")  
                    continue                                                    # Skip the current repetition and wait for new frames

                # Convert the received color frame into an image array
                color_image = np.asanyarray(color_frame.get_data())             # Convert the color frame to a NumPy array (for calculations)
                depth_image = np.asanyarray(depth_frame.get_data())             # Convert the depth frame to a NumPy array (for calculations)

                # Preprocess the color image
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)            # Convert the color image to grayscale
                blurred = cv2.GaussianBlur(gray, (1, 1), 0)                     # Apply a Gaussian blur to reduce noise (blurring image)
                edges = cv2.Canny(blurred, lower_threshold, upper_threshold)    # Perform Canny edge detection (detecting the edges)

                # Detect contours in the edge-detected image
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Extract external contours
                if contours:                                                    # Check if contours are found 
                    largest_contour = max(contours, key=cv2.contourArea)        # Find the largest contour 
                    (x, y), radius = cv2.minEnclosingCircle(largest_contour)    # Get the center and radius of the minimum enclosing circle for the largest contour

                    vertical_x = int(x)                                         # Convert the x-coordinate to an integer
                    start_point = min(largest_contour, key=lambda p: abs(p[0][0] - vertical_x))     # Find the point closest to the vertical line 
                    start_index = np.where((largest_contour == start_point).all(axis=2))[0][0]      # Find the index of the start point in the contour
                    reordered_contour = np.concatenate((largest_contour[start_index:], largest_contour[:start_index]))  # Reorder the contour to start from the identified start point

                    circle_area = np.pi * (radius ** 2)                         # Calculate the area of the minimum enclosing circle
                    contour_area = cv2.contourArea(largest_contour)             # Calculate the area of the contour
                    if 0.7 < contour_area / circle_area < 1.0:                  # Check if the contour is circular
                        depth_at_center = depth_frame.get_distance(int(x), int(y))                  # Get the depth (distance)

                        if depth_at_center > 0:  
                            pixels_per_mm = pixels_per_mm_at_reference_distance * (reference_distance_m / depth_at_center)  # Adjust pixels per mm
                            diameter_mm = (2 * radius) / pixels_per_mm          # Calculate the diameter of the gear in millimeters

                            if min_diameter_mm <= diameter_mm <= max_diameter_mm:  # Check if the diameter is within the acceptable range
                                center_x_mm = x / pixels_per_mm                 # Convert the x-coordinate of the center to millimeters
                                center_y_mm = y / pixels_per_mm                 # Convert the y-coordinate of the center to millimeters

                                send_status_to_twincat("Gear found", plc)
                                set_restart_variable_true(plc, variable_name="Main.restart", value=True)                

                                ## Draw the contour and comments in the final image
                                cv2.drawContours(color_image, [reordered_contour], -1, (0, 255, 0), 2)                  # Draw the contour in green
                                #cv2.circle(color_image, (int(x), int(y)), int(radius), (255, 0, 0), 2)                 # Draw the circle around the contour (commented out)
                                #cv2.line(color_image, (int(x), 0), (int(x), color_image.shape[0]), (255, 0, 0), 2)     # Draw vertical line through the center (commented out)
                                cv2.putText(color_image, "X-axis", (int(x) + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)                            # Label X-axis
                                #cv2.line(color_image, (0, int(y)), (color_image.shape[1], int(y)), (0, 255, 0), 2)     # Draw horizontal line through the center (commented out)
                                cv2.putText(color_image, "Y-axis", (color_image.shape[1] - 100, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)    # Label Y-axis

                                ## Create lists for the coordinates of the contour in mm and angle in degrees
                                x_coords_mm = []
                                y_coords_deg = []
                                for point in reordered_contour:                 # Repeat over each point in the reordered contour
                                    pixel_x, pixel_y = point[0]                 # Extract the pixel coordinates
                                    mm_x = (pixel_x - x) / pixels_per_mm        # Convert the x-coordinate to millimeters
                                    mm_y = (pixel_y - y) / pixels_per_mm        # Convert the y-coordinate to millimeters

                                    distance = np.sqrt(mm_x**2 + mm_y**2)       # Calculate the distance from the center
                                    angle_rad = np.arctan2(mm_y, mm_x)          # Calculate the angle in radians
                                    angle_deg = np.degrees(angle_rad)           # Convert the angle to degrees
                                    angle_deg = (angle_deg + 90) % 360          # Adjust the angle so that it start at 0 degrees

                                    x_coords_mm.append(distance)                # Add the distance to the x-coordinates list
                                    y_coords_deg.append(angle_deg)              # Add the angle to the y-coordinates list

                                ## Determine the angle of the first point
                                first_point_x, first_point_y = start_point[0]   # Get the coordinates of the first point 
                                first_angle_rad = np.arctan2(first_point_y - y, first_point_x - x)                      # Calculate the angle of the first point
                                first_angle_deg = np.degrees(first_angle_rad)   # Convert the angle to degrees
                                first_angle_deg = (first_angle_deg + 90) % 360  # Adjust the angle to be 0 at the top

                                ## Shift the contour to 0 degrees if it's near 180 degrees
                                if 170 <= first_angle_deg <= 190:
                                    angle_shift = 180                           # Shift the angle by 180 degrees 
                                    y_coords_deg = [(angle + angle_shift) % 360 for angle in y_coords_deg]              # Apply the angle shift to all angles

                                ## Write the coordinates to a CSV file
                                with open(csv_filename, mode="w", newline="") as file:
                                    writer = csv.writer(file)                   # Open the CSV file 
                                    writer.writerow(["Distance (mm)", "Angle (degrees)"])                               # Write distance and degrees to csv header
                                    for x, y in zip(x_coords_mm, y_coords_deg): # Write the distance and angle for each point
                                        writer.writerow([x, y])

                                send_status_to_twincat("Sending coordinates", plc)  
                                send_coordinates_to_twincat(x_coords_mm, y_coords_deg)  
                                send_status_to_twincat("Coordinates successfully sent", plc)  

                                ## Mark the first point and adjust the angle for visualization
                                cv2.circle(color_image, (first_point_x, first_point_y), 3, (0, 0, 255), -1)             # Draw a red circle at the first point
                                cv2.putText(color_image, "First Point", (first_point_x + 10, first_point_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Label the first point


                                ## Image the result in PNG
                                if x_coords_mm:                                 # Check the x_coords_mm list 
                                    first_x = x_coords_mm[0]                    # Get the first x-coordinate 
                                    first_angle = y_coords_deg[0]               # Get the first angle

                                    # Determine the pixel coordinates of the first point for drawing
                                    first_point_x = int(x + first_x * pixels_per_mm)
                                    first_point_y = int(y)

                                    # Display the first coordinates and angle on the image
                                    cv2.putText(color_image, f"First Coordinates: {first_x:.2f} mm, Angle: {first_angle:.2f} degrees",
                                                (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                                # Display the calculated diameter of the gear on the image
                                cv2.putText(color_image, f"Diameter: {diameter_mm:.2f} mm",
                                            (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                                # Save the image
                                cv2.imwrite("Result_gear.png", color_image)     # Save the result image to a file
                                
                                # Stop processing here
                                send_status_to_twincat("Process completed", plc)  
                                send_next_move_to_twincat("Ready for new input", plc) 
                                print("Processing completed.")  
                                processing = False                              # Set processing to False, awaiting the next pulse
                                set_restart_variable_false(plc, variable_name="Main.restart", value=False) 
                                set_process_completed_true(plc, variable_name="Main.coordinatesreceived", value=True) 
                                break                                           # Exit the current processing loop and restart the main loop

                            else:
                                print(f"Object: diameter ({diameter_mm:.2f} mm) is not between {min_diameter_mm} mm and {max_diameter_mm} mm.")  # Log if the diameter is out of range
                        else:
                            print("Object not circular") 
                else:
                    print("No contours found")  
                
    except KeyboardInterrupt:                                                   # Manual interruption of the program
        print("Program interrupted manually.")  
    finally:
        pipeline.stop()                                                         # Stop the camera pipeline
        plc.close()                                                             # Close the connection to the PLC
        cv2.destroyAllWindows()                                                 # Close any OpenCV windows

## Main function to run this script
if __name__ == "__main__":
    main()                                                                      # Execute the main function

