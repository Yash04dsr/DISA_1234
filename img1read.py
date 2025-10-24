import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load the Image ---
# Load the image from the specified path
image_path = 'img2.png'
image = cv2.imread(image_path)
# Create a copy to draw the final contour on
output_image = image.copy()

# --- 2. Preprocessing (Applying Filters) ---
# Convert the image to grayscale for easier processing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian Blur to reduce noise and improve edge detection
# The (7, 7) kernel size can be adjusted. Larger means more blur.
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# --- 3. Edge Detection ---
# Use the Canny edge detector to find outlines
# The two threshold values (50, 150) can be tuned.
# Edges with intensity gradient > 150 are sure to be edges (high threshold).
# Edges with intensity gradient < 50 are sure to be non-edges (low threshold).
# Edges between the two are accepted only if they are connected to a "sure" edge.
canny_edges = cv2.Canny(blurred, 50, 150)

# --- 4. Find and Filter Contours ---
# Find contours in the edge-detected image
# cv2.RETR_EXTERNAL retrieves only the extreme outer contours.
# cv2.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments.
contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Check if any contours were found
if contours:
    # Find the largest contour by area, assuming it's the droplet
    main_contour = max(contours, key=cv2.contourArea)
    
    # --- 5. Draw the Contour ---
    # Draw the largest contour on the original image copy
    # A green line with a thickness of 2 pixels is used here
    cv2.drawContours(output_image, [main_contour], -1, (0, 255, 0), 2)

# --- 6. Display the Results ---
# Use matplotlib to show the steps and the final result
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Canny Edges')
plt.imshow(canny_edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Detected Contour')
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()