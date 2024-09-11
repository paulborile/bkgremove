import cv2
import numpy as np

# Load the image
image = cv2.imread('medium.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
edges = cv2.Canny(image, 100, 200)

# Find contours from the edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create an empty mask
mask = np.zeros_like(image)

# Fill the detected contours (foreground) on the mask
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

# Optional: Refine the mask
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Apply the mask to the original image (optional, just for visualization)
result = cv2.bitwise_and(image, image, mask=mask)

# Save or display the result
cv2.imwrite('foreground_mask.jpg', mask)
cv2.imshow('Mask', mask)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

