import cv2
import matplotlib.pyplot as plt

# Laad de afbeelding met ArUco-markers
image = cv2.imread('Aruco.png')

# Converteer de afbeelding naar grijswaarden
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Definieer het ArUco dictionary type
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)  # Dit werkt voor de meeste versies

# Initialiseer de parameters voor de ArUco detector
parameters = cv2.aruco.DetectorParameters_create()

# Detecteer de ArUco-markers in de afbeelding
corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

# Als markers gevonden zijn
if ids is not None:
    # Teken de randen van de ArUco-markers
    cv2.aruco.drawDetectedMarkers(image, corners, ids)

    # Toon de afbeelding met gedetecteerde ArUco-markers
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Gedetecteerde ArUco Markers')
    plt.axis('off')
    plt.show()
else:
    print("Geen ArUco-markers gevonden.")
