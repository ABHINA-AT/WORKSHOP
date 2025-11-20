import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load an cvtColorimage
image = cv2.imread(r"C:\Users\user\OneDrive\Desktop\WORKSHOP\pexels-chevanon-1108099.jpg")
# Replace with your image path
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(image)
plt.axis('off')
plt.show()
# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Display results
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(image)
# Apply Canny Edge Detection
edges = cv2.Canny(gray, threshold1=50, threshold2=150)

# Display results
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('gray_scale')
plt.imshow(gray)

plt.subplot(1,2,2)
plt.title('Edge Detection')
plt.imshow(edges, cmap='gray')

plt.show()
