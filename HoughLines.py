"""Line Detection using Canny Edge Detection and Hough Transform with OpenCV"""

#Using the Hough transform, detect and draw lines in the image "autocesta.jpeg".
#Using the image masking method, try to extract only the train lines tape. 
#It is recommended that you use the upper half of the ellipse in the lower half of the image as a mask.

import cv2
import numpy as np

#Loading and Preprocessing the Image
img = cv2.imread("data/autocesta.jpeg")

img_canny = cv2.Canny(img, 90, 200)

#Creating an Elliptical Mask
maska_elipse = np.zeros_like(img_canny)
height, widht = img_canny.shape 

cv2.ellipse (maska_elipse,
            (widht //2, height),
            axes = (widht //2, height//2), 
            angle = 0,
            startAngle = 180,
            endAngle=360,
            color=255,
            thickness=-2)

#Applying the Elliptical Mask
elipsa_filtered = cv2.bitwise_and(img_canny, img_canny, mask= maska_elipse)

#Detecting and Drawing Lines
lines = cv2.HoughLinesP(elipsa_filtered, 2, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
lines = lines.squeeze()

for line in lines:
    cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0,255,0), thickness=2)
    
#Displaying Images
cv2.imshow("Original image", img)
cv2.imshow("Canny image", img_canny)
cv2.imshow("Detected lines", elipsa_filtered)
cv2.waitKey()
cv2.destroyAllWindows()