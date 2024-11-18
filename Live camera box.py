import cv2
import numpy as np

# Functie om de grootste contour te vinden
def get_largest_contour(contours):
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        return max_contour
    return None

# Start de video capture (0 is de default camera)
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Kan de camera niet openen")
    exit()

while True:
    # Lees een frame van de camera
    ret, frame = cap.read()
    
    if not ret:
        print("Kan geen frame ophalen")
        break
    
    # Omzetten naar grijswaarden en drempelbepaling uitvoeren
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Contouren detecteren
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Vind de grootste contour (tandwiel) en bereken diameter
    largest_contour = get_largest_contour(contours)
    
    if largest_contour is not None:
        # Omcirkelende cirkel bepalen
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        diameter = 2 * radius

        # Teken de contour en de cirkel op het frame
        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
        cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
        
        # Weergeef de diameter op het frame
        cv2.putText(frame, f"Diameter: {diameter:.2f} pixels", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Toon het frame
    cv2.imshow('Tandwiel met Diameter', frame)

    # Stoppen met 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Sluit alles af
cap.release()
cv2.destroyAllWindows()
