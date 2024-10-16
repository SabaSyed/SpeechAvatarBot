from rembg import remove
import numpy as np
import cv2

# Read the input image
input_path = 'img2.jpeg'  # Replace with your image path
output_path = 'output_image.png'

# Open the image using OpenCV
input_image = cv2.imread(input_path)

# Remove the background
output_image = remove(input_image)

# Save the output image
cv2.imwrite(output_path, output_image)

print(f"Background removed. Output saved at: {output_path}")
