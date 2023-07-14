"""Object Tracking with MeanShift Algorithm"""

import cv2
import numpy as np

#Opening Video Capture
capture = cv2.VideoCapture("data/red_car_moving.mp4")
ret, frame =capture.read()

#Video Capture and Initialization
x, y, w, h = 233,202,34,20
track_window = (x,y,w,h)

#ROI Initialization
roi = frame[y:y+h, x:x+w]
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_mask = cv2.inRange(roi_hsv, np.array([170, 50, 50]), np.array([180, 255, 255])) #funkcija vraca binarnu sliku, sto uzme maska bit ce 255(bijelo) po defaultu

roi_hist = cv2.calcHist([roi_hsv], [0], roi_mask, [180], [0,180])
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

#MeanShift Tracking Loop
term_criteria = (cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, 10, 1)

while True:
    ret, frame = capture.read()
    if ret:
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_projected_img = cv2.calcBackProject([frame_hsv], [0], roi_hist, [0,180], 1)
        ret, track_window = cv2.meanShift(back_projected_img, track_window, term_criteria)
        cv2.rectangle(frame, track_window, color= (0,255,0), thickness= 2)
        cv2.imshow("Detection",frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break
    
#Cleaning Up
capture.relase()
cv2.destroyAllWindows()


