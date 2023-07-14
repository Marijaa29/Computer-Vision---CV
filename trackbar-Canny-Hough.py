"""Interactive Color and Edge Filtering with Trackbars"""

"""
1. The algorithm should work on video
2. Pause the video on the "p" key (the rest of the algorithm must continue to be executed on the paused frame)
3. Rework the algorithm to display images in multiple rows (eg 2 images in one row) (cv2.vconcat())
4. Add trackbars for canny thresholds and display canny image (cv2.Canny())
5. Add a feature that you think would be useful

"""
import cv2
import numpy as np

# Trackbar Initialization
cv2.namedWindow("filter")

low = np.array([0, 0, 0])
high = np.array([255, 255, 255])

is_paused = False

canny_low = 100
canny_high = 200

# Trackbar Callback Functions
def first_low_trackbar(val):
    global low
    global high
    low[0] = val
    high[0] = max(high[0], low[0]+1)
    cv2.setTrackbarPos("High-0", "filter", high[0])

def first_high_trackbar(val):
    global low
    global high
    high[0] = val
    low[0] = min(high[0]-1, low[0])
    cv2.setTrackbarPos("Low-0", "filter", low[0])

def second_low_trackbar(val):
    global low
    global high
    low[1] = val
    high[1] = max(high[1], low[1]+1)
    cv2.setTrackbarPos("High-1", "filter", high[1])

def second_high_trackbar(val):
    global low
    global high
    high[1] = val
    low[1] = min(high[1]-1, low[1])
    cv2.setTrackbarPos("Low-1", "filter", low[1])

def third_low_trackbar(val):
    global low
    global high
    low[2] = val
    high[2] = max(high[2], low[2]+1)
    cv2.setTrackbarPos("High-2", "filter", high[2])

def third_high_trackbar(val):
    global low
    global high
    high[2] = val
    low[2] = min(high[2]-1, low[2])
    cv2.setTrackbarPos("Low-2", "filter", low[2])
    
def canny_low_trackbar(val):
    global canny_low
    canny_low = val


def canny_high_trackbar(val):
    global canny_high
    canny_high = val

def pause_video():
    global is_paused
    is_paused = not is_paused
    

cv2.createTrackbar("Low-0", "filter", 0, 255, first_low_trackbar)
cv2.createTrackbar("High-0", "filter", 255, 255, first_high_trackbar)
cv2.createTrackbar("Low-1", "filter", 0, 255, second_low_trackbar)
cv2.createTrackbar("High-1", "filter", 255, 255, second_high_trackbar)
cv2.createTrackbar("Low-2", "filter", 0, 255, third_low_trackbar)
cv2.createTrackbar("High-2", "filter", 255, 255, third_high_trackbar)
cv2.createTrackbar("Canny Low", "filter", 100, 255, canny_low_trackbar)
cv2.createTrackbar("Canny High", "filter", 200, 255, canny_high_trackbar)

# Video Capture and Filter Application Loop
cap = cv2.VideoCapture("data/video1.mp4")

while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (200, 200))

            # RGB filter
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            filteredRGB = cv2.inRange(frameRGB, low, high)
            filteredRGB = cv2.cvtColor(filteredRGB, cv2.COLOR_GRAY2BGR)

            # HSV filter
            frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            filteredHSV = cv2.inRange(frameHSV, low, high)
            filteredHSV = cv2.cvtColor(filteredHSV, cv2.COLOR_GRAY2BGR)

            # HLS filter
            frameHLS = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
            filteredHLS = cv2.inRange(frameHLS, low, high)
            filteredHLS = cv2.cvtColor(filteredHLS, cv2.COLOR_GRAY2BGR)

            # Canny filter
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            canny_edges = cv2.Canny(frameGray, canny_low, canny_high)
            canny_edges = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
            
            
            showIMG = cv2.vconcat([frame, filteredRGB, filteredHLS])
            showIMG2 = cv2.vconcat([frame, filteredHSV, canny_edges])
            showIMGx = cv2.hconcat([showIMG, showIMG2])
            cv2.imshow("filterq", showIMGx)
        
        # Keyboard Input Handling
        key = cv2.waitKey(80)

        if key == ord("q"):
            print(f"LOW: {low} | HIGH: {high}")
            break
        elif key == ord("p"):
            is_paused = not is_paused

# Release Resources
cap.release()
cv2.destroyAllWindows()
