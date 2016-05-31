#Script to capture images and save for training/validation.
#When video screen is visible, press l to set the label and r to begin and end capture

import handRecognition as hr
import cv2 as cv
from sklearn import svm
from sklearn import preprocessing
from sklearn.externals import joblib

count = 0
labels = []
samples = []
capture_flag = False
cap = cv.VideoCapture(0)

while True:
    _, im = cap.read()        
    key = cv.waitKey(1)
    segIm = hr.segment(im)
    numFingers, fingerSum, longestDist = hr.extract(im,segIm)
    
    #quit if q pressed
    if key == ord('q'):
        cap.release()
        cv.destroyAllWindows()
        break   
        
    if key == ord('l'):
        current_label = raw_input('Current label: ')
    
    if key == ord('r'):
        if capture_flag:
            capture_flag = False
            print "samples for",current_label,": ",count
        else:
            print "capturing..."
            capture_flag = True
    
    if capture_flag:
        cv.imwrite('image_' + str(count) + '.jpeg' ,im)
        sampleData = [numFingers, fingerSum, longestDist]
        labels.append(current_label)
        samples.append(sampleData)
        count += 1

#Write images labels to a file
fo = open('image_labels', 'w')
fo.write(str(labels))
fo.close()

#Save scaling info to copy into the classify function of handRecognition.py
scaler = preprocessing.StandardScaler().fit(samples)
fo = open("data_info.txt","w")
fo.write("Data mean: {}\n".format(scaler.mean_))
fo.write("Data std: {}".format(scaler.scale_))
fo.close()

#Train and save support vector machine
model = svm.SVC()
model.fit(scaler.transform(samples),labels)
joblib.dump(model,'gesture_svm.pkl')

