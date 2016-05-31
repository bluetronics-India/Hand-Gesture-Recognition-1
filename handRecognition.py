# Functions to perform hand gesture recognition

import math
import numpy as np
import cv2 as cv

#####################################
# Distance Formula Function
#####################################

def dist(x1, x2, y1, y2):
    sq1 = (x1 - x2) * (x1 - x2)
    sq2 = (y1 - y2) * (y1 - y2)
    return math.sqrt(sq1 + sq2)


######################################
# Segmentation function
######################################
def segment(im):
    # create grayscale image
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    # create detrended image by approximating trend for removal with wide gaussian kernel
    trend = cv.GaussianBlur(gray, (377, 391), 0) - 10
    detrended = gray - trend

    # Use Otsu's method to find an appropriate threshold
    _, otsu = cv.threshold(detrended, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    otsu = cv.morphologyEx(otsu, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7)))

    # HSV color based segmentation
    hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    lower = np.array([8, 55, 0])
    upper = np.array([31, 155, 255])
    hsv = cv.GaussianBlur(hsv, (5, 5), 0)
    hsvSeg = cv.inRange(hsv, lower, upper)
    hsvSeg = cv.morphologyEx(hsvSeg, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7)))

    # Watershed based segmentation
    # Finding sure background area
    bg = cv.dilate(hsvSeg, None, iterations=2)
    ret, bg = cv.threshold(bg, 1, 128, 1)

    # Finding sure foreground area
    fg = cv.erode(hsvSeg, None, iterations=1)

    # Create marker for watershed
    marker = np.int32(cv.add(bg, fg))

    # Watershed image
    wtrShed = cv.convertScaleAbs(cv.watershed(im, marker))
    wtrShed[wtrShed <= 128] = 0

    # final voting, hsvSeg double weighted
    otsu = otsu / 255
    hsvSeg = hsvSeg * (2 / 255)
    wtrShed = wtrShed / 255
    finalIm = otsu + hsvSeg + wtrShed
    finalIm[finalIm == 0] = 0
    finalIm[finalIm >= 1] = 255
    return finalIm


############################################
# Feature Extraction Function
############################################

def extract(im, segIm):
    # Get contours and find the largest one (presumably outlining the hand)
    contoursIm, contoursList, hierarchy = cv.findContours(segIm.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    if len(contoursList) == 0:
        return [], [], []
    maxArea = 0
    maxIndex = 0
    for i in range(len(contoursList)):
        area = cv.contourArea(contoursList[i])
        if area > maxArea:
            maxIndex = i
            maxArea = area

    cv.drawContours(im, [contoursList[maxIndex]], 0, (0, 255, 0), 0)

    # Now that we have the contours, find the convex points of the contour
    hull = cv.convexHull(contoursList[maxIndex], returnPoints=False)
    defects = cv.convexityDefects(contoursList[maxIndex], hull)

    # Get moment of the contours (to approximate the palm location)
    moment = cv.moments(contoursList[maxIndex])
    xMoment = int(moment['m10'] / moment['m00'])
    yMoment = int(moment['m01'] / moment['m00'])
    cv.circle(im, (xMoment, yMoment), 10, (255, 0, 0), -1)

    # Pull fingers from the contours, checking for duplicates based on close distances
    fingersPos = [];
    fingerLen = [];
    firstFlag = True
    for i in range(defects.shape[0]):
        s, _, _, d = defects[i, 0]
        if d / 256 > 18:  # elimnate if too small
            start = tuple(contoursList[maxIndex][s][0])

            # always add first defect
            if firstFlag and start[1] < 350:
                firstFlag = False
                fingersPos.append(start)
                fingerLen.append(dist(xMoment, start[0], yMoment, start[1]))
                cv.circle(im, start, 5, [0, 0, 255], -1)
                cv.line(im, (xMoment, yMoment), start, (255, 0, 0), 5)

            # Check all recorded defects to see if they are far enough from other defects to be a different finger
            addFingertip = False
            for knownFinger in fingersPos:
                distance = dist(start[0], knownFinger[0], start[1], knownFinger[1])
                if distance > 50 and start[1] < 350:
                    addFingertip = True
                    cv.circle(im, start, 5, [0, 0, 255], -1)
                    cv.line(im, (xMoment, yMoment), start, (255, 0, 0), 5)
            if addFingertip:
                fingersPos.append(start)
                fingerLen.append(dist(xMoment, start[0], yMoment, start[1]))

    # Normalized FingerLength Feature
    numFingers = (len(fingerPos) - 3.18181818) / 1.05038364

    # Normalized FingerSum Feature
    fingerSum = 0
    for length in fingerLen:
        fingerSum = fingerSum + length
    fingerSum = (fingerSum - 517.38688929) / 187.34377504

    # Normalized longest X distance between fingertips feature
    shortIndex = 0
    longIndex = 0
    n = 0
    for finger in fingerPos:
        if finger[0] < fingerPos[shortIndex][0]:
            shortIndex = n
        if finger[0] > fingerPos[longIndex][0]:
            longIndex = n
        n = n + 1
    longestDist = dist(fingerPos[shortIndex][0], fingerPos[longIndex][0], fingerPos[shortIndex][1],
                       fingerPos[longIndex][1])
    longestDist = (longestDist - 158.18776099) / 97.47535787

    return numFingers, fingerSum, longestDist


####################################
# Feature Classification Function
####################################

def classify(numFingers, fingerSum, longestDist, im, model):
    sampleFeatures = np.array([numFingers, fingerSum, longestDist])
    sampleFeatures.reshape(1, -1)
    cv.putText(im, "Hand Gesture: {}".format(model.predict(sampleFeatures)[0]), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1,
               255)
    cv.imshow("im", im)
