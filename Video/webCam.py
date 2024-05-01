import cv2
from rembg import remove
import numpy as np
img = cv2.imread("./data/panda2.png", cv2.IMREAD_GRAYSCALE)  # queryiamge
img = remove(img)
cap = cv2.VideoCapture(0)

# Features
sift = cv2.SIFT_create()
kp_image, desc_image = sift.detectAndCompute(img, None)
# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:
    _, frame = cap.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage

    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)

    good_points = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_points.append(m)
    #img3 = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)
    img3 = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, None,
                           flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow("img3", img3)
    #cv2.imshow("grayFrame", grayframe)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
