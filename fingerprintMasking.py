"""Fingerprint Masking"""

import cv2
import numpy as np

#Reading and Thresholding the Image
img_gray = cv2.imread("data/otisak_prsta.png", cv2.IMREAD_GRAYSCALE)

_, img_binary = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

#print(img_binary.shape)
#print(img_binary.dtype)

#Creating the Mask
mask = np.zeros((512,512), dtype=np.uint8)
height, width = img_gray.shape

#Defining Mask Points
x1 = width // 4
y1 = height -1
x2 = 0
y2 = 3 * (height // 4)
x3 = 0
y3 = height // 4
x4 = width // 4
y4 = 0
x5 = 3 * (width // 4)
y5 = 0
x6 = width -1
y6 = height // 4
x7 = width -1
y7 = 3 * (height // 4)
x8 = 3 * (width // 4)
y8 = height -1

#Filling the Mask with Points
mask_points = np.array([[[x1,y1], [x2,y2], [x3,y3], [x4,y4], [x5,y5], [x6,y6], [x7,y7], [x8,y8]]])

mask = cv2.fillPoly(mask, mask_points, 255)

#Applying Mask to the Binary Image
masked_image = cv2.bitwise_and(img_binary, mask)

#Displaying the Results
cv2.imshow("Original-gray", img_gray)
cv2.imshow("Mask", mask)
cv2.imshow("Masked image", masked_image)
cv2.waitKey()
cv2.destroyAllWindows()