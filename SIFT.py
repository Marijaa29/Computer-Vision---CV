"""Feature Matching and Object Detection using SIFT"""

import cv2
import numpy as np


#Constants and Input Images
MIN_MATCH_COUNT = 4

img_query = cv2.imread("data/car.png", cv2.IMREAD_GRAYSCALE)
img_train = cv2.imread("data/dashcam_first_frame.png", cv2.IMREAD_GRAYSCALE)

#SIFT Feature Extraction
sift = cv2.SIFT_create()

kp_query, desc_query = sift.detectAndCompute(img_query, None)
query_kp_img = cv2.drawKeypoints(img_query, kp_query, None)

kp_train, desc_train =sift.detectAndCompute(img_train, None)
train_kp_img = cv2.drawKeypoints(img_train, kp_train, None)

#Feature Matching
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc_query, desc_train, k=2)
img_matches = cv2.drawMatchesKnn(img_query, kp_query, img_train, kp_train, matches, None)

#Good Matches Filtering
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
        
img_good_matches = cv2.drawMatchesKnn(img_query, kp_query, img_train, kp_train, [good_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 

#Homography and Object Detection
if len(good_matches) >= MIN_MATCH_COUNT:
    src_points = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_points = np.float32([kp_train[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
    h,w = img_query.shape
    query_bb_points = np.float32([[0,0],[0, h-1],[w-1, h-1],[w-1, 0]]).reshape(-1,1,2)
    train_bb_points = cv2.perspectiveTransform(query_bb_points, M)
    img_detection = cv2.polylines(img_train, [np.int32(train_bb_points)], True, 255, thickness=2)
else:
    print("Not enough matching descriptors found - {} / {}".format(len(good_matches), MIN_MATCH_COUNT))

#Displaying Results
cv2.imshow("Query keypoints", query_kp_img)
cv2.imshow("Train keypoints", train_kp_img)
cv2.imshow("Matches", img_matches)
cv2.imshow("Good matches", img_good_matches)
cv2.imshow("Detection", img_detection)
cv2.waitKey()
cv2.destroyAllWindows()
