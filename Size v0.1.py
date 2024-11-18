import cv2

# Laad de afbeelding
image = cv2.imread('Big_gear flits v0.1.jpg')

# Verkrijg de afmetingen in pixels
height, width, _ = image.shape

# Stel de PPI (pixels per inch) in; meestal tussen 72 en 300 voor afbeeldingen
ppi = 477*2  # Pas dit aan op basis van je afbeelding
#460 voor een iPhone 13
#477 voor de ArduCam IMX477 12MP
# Bereken de afmetingen in millimeters
width_mm = (width / ppi) * 25.4
height_mm = (height / ppi) * 25.4

# Toon de afmetingen in mm
print(f"Afmetingen van de afbeelding: {width_mm:.2f} mm x {height_mm:.2f} mm")


# Small gear = 68mm
# Big gear = 102mm