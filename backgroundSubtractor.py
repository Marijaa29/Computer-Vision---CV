"""Video Background Subtraction using OpenCV"""

import cv2
import numpy as np

#Opening Video Capture
capture = cv2.VideoCapture("data/red_car_moving.mp4")

#Checking Video Opening Status and Getting FPS
if not capture.isOpened():
    print("Error: Cannot open video")
    exit(1)

print(capture.get(cv2.CAP_PROP_FPS)) 

#Creating Background Subtractor 
backSub = cv2.createBackgroundSubtractorMOG2()

#Processing Video Frames

while True:
    ret, frame = capture.read()
    
    if frame is None:
        break
    maska = backSub.apply(frame)
    cv2.imshow("Frame", frame)
    cv2.imshow("Output", maska)
    if cv2.waitKey(1) in [ord("q"), ord("Q")]:
        break

#Releasing Resources
capture.release()
cv2.destroyAllWindows()
