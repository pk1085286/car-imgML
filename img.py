import cv2
import numpy as np

# Load the car image
car_image = cv2.imread('car_image.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to reduce noise in the image
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Perform Canny edge detection to detect edges in the image
canny_image = cv2.Canny(blurred_image, 50, 150)

# Apply a binary threshold to the image to separate the car from the background
ret, threshold_image = cv2.threshold(canny_image, 127, 255, cv2.THRESH_BINARY)

# Find contours in the image
contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
cv2.drawContours(car_image, contours, -1, (0, 255, 0), 2)

# Display the processed image
cv2.imshow('Processed Car Image', car_image)
cv2.waitKey(0)
