""""Color-Based Stop Sign Detection"""

#Load the image “stop_sign.jpg”. Use the color filtering method to detect the stop sign.

import cv2
import numpy as np


#Reading and Preprocessing the Image
image = cv2.imread("data/stop_sign.jpg")

image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#Defining Color Boundaries
lower_red_boundary1 = np.array([0,160,100]) 
upper_red_boundary1 = np.array([15,255,255]) 
lower_red_boundary2 = np.array([165,160,100])
upper_red_boundary2 = np.array([179,255,255])

#Creating the Mask
red_mask = cv2.inRange(image_hsv, lower_red_boundary1, upper_red_boundary1)
red_mask2 = cv2.inRange(image_hsv, lower_red_boundary2, upper_red_boundary2)
maska = cv2.bitwise_or(red_mask, red_mask2)

#Applying Morphological Operations
kernel = np.ones((3,3), np.uint8)
maska = cv2.morphologyEx(maska, cv2.MORPH_OPEN, kernel)

#Filtering the Image
red_mask_filtered = cv2.bitwise_and(image, image, mask=maska)

#Drawing the Bounding Rectangle
rec = cv2.boundingRect(maska)
cv2.rectangle(image, rec, color = [0,255,0], thickness=2)

#Displaying Results
cv2.imshow("Original image", image)
cv2.imshow("Mask filltered", red_mask_filtered)
cv2.waitKey()
cv2.destroyAllWindows()