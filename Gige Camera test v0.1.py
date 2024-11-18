import cv2

camera_ip = "rtsp://192.168.1.110/live"  # Adjust based on your camera's correct RTSP stream URL

# Open the video stream
cap = cv2.VideoCapture(camera_ip)

# Increase the timeout (in milliseconds)
cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)  # 60 seconds timeout

if not cap.isOpened():
    print("Error: Unable to open the GigE camera stream.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to grab frame from GigE camera.")
            break

        # Display the frame
        cv2.imshow("GigE Camera Stream", frame)

        # Press 'q' to quit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
