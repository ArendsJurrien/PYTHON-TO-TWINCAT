import cv2
import numpy as np

# Functie om de grootste contour (omtrek van het tandwiel) te vinden
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

    # Zet het beeld om naar grijswaarden en voer drempelbepaling uit
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Contouren detecteren
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Vind de grootste contour (aangenomen het tandwiel)
    largest_contour = get_largest_contour(contours)

    if largest_contour is not None:
        # Teken het pad van de omtrek van het tandwiel
        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

        # Weergeef de omtrek als pad op het frame
        cv2.putText(frame, "Tandwielomtrek gedetecteerd", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    else:
        cv2.putText(frame, "Geen tandwiel gevonden", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Toon het frame met het pad van de omtrek
    cv2.imshow('Tandwielomtrek', frame)

    # Stoppen met 'q' en sla het beeld op
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Sla de afbeelding op met de getekende omtrek
        cv2.imwrite("gear_contour.jpg", frame)
        print("Afbeelding opgeslagen: gear_contour.jpg")
        break

# Sluit alles af
cap.release()
cv2.destroyAllWindows()
