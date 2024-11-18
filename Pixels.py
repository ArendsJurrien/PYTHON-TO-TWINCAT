import cv2
import numpy as np
import matplotlib.pyplot as plt

# Laad de afbeelding
image = cv2.imread('Big_gear flits v0.2.jpg', cv2.IMREAD_GRAYSCALE)

# Bepaal een doorsnede (bijv. de 100e rij)
row = 100
horizontal_slice = image[row, :]

# Plot de pixelintensiteiten langs de rij
plt.figure(figsize=(10, 5))
plt.plot(horizontal_slice, label=f'Row {row}')
plt.title(f'Pixel Intensities Along Row {row}')
plt.xlabel('Column Index')
plt.ylabel('Pixel Intensity')
plt.legend()
plt.show()