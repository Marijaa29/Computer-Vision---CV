""" "Shape Detection and Line Drawing"""

import cv2
import numpy as np

#Reading and Blurring the Image
img = cv2.imread("data/oblici.png")
img_blurred = cv2.blur(img, (3,3))

#Performing Canny Edge Detection
canny_image= cv2.Canny(img, 100 ,200)

#Applying Hough Transform for Line Detection
lines=cv2.HoughLinesP(canny_image, 1, np.pi /180, 70, minLineLength=20,maxLineGap=5)
lines = lines.squeeze()

#Drawing Detected Lines on the Image
for line in lines:
    cv2.line(img,(line[0],line[1]),(line[2],line[3]),(0,0,255),thickness=2) 

#Drawing Detected Lines on the Image
cv2.imshow("Image",img)
cv2.waitKey()
cv2.destroyAllWindows()