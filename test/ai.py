import cv2
import numpy as np
from matplotlib import pyplot as plt


# Funkcja do przekształcenia konturów na punkty w macierzy 20x20
def map_contour_to_grid(contour):
    contour = np.squeeze(contour)
    contour = contour.astype(np.float32)
    contour[:, 0] = contour[:, 0] * (19 / np.max(contour[:, 0]))
    contour[:, 1] = contour[:, 1] * (19 / np.max(contour[:, 1]))
    contour = np.round(contour).astype(np.int32)
    
    grid = np.zeros((20, 20))
    for point in contour:
        grid[point[1], point[0]] = 255
    
    return grid

# Tworzenie planszy 20x20
board = np.zeros((20, 20))

# Wczytanie obrazu
img = cv2.imread('shapes.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Iterowanie po konturach
for contour in contours:
    if len(contour) < 5:
        continue
    
    # Przekształcenie konturu na macierz 20x20
    grid = map_contour_to_grid(contour)
    
    # Dodanie konturu do planszy
    board = np.maximum(board, grid)

# Wyświetlenie planszy
plt.imshow(board, cmap='gray')
plt.show()
