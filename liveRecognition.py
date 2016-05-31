#Charlie Osborne and Brent Marin
#3/24/16
#Hand Gesture Recognition Project
#Main script for live recognition

import handRecognition as hr
import cv2 as cv
from sklearn.externals import joblib

#Load trained svm
model = joblib.load('gesture_svm.pkl')

#open video camera, 0 is default camera on the computer
cap = cv.VideoCapture(0)

while(True):
    _, im = cap.read()
    segIm = hr.segment(im)
    numFingers, fingerSum, longestDist = hr.extract(im,segIm)
    hr.classify(numFingers, fingerSum, longestDist, im, model)

    #q key press will end video capture
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
