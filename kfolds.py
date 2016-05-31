# Charlie Osborne and Brent Marin
# K-folds cross validation test

import handRecognition as hr
import numpy as np
import cv2 as cv
from sklearn import svm
from sklearn import preprocessing

# Image labels from image_labels returned by build_database.py
label_array = ['open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'peace', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'shakka', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down', 'two down']

# open video camera, 0 is default camera on the computer
samples = []
labels = []
current_label = ''
capture_flag = False
count = 0
accuracy_array = []

#build 21 subsets of 20 images
arr = np.arange(420)
np.random.shuffle(arr)
sets = np.array_split(arr, 21)

#Run k-cross folds validation, 21 iterations
for i in range(21):
    print '''
    ###############################################
    
    Current Iteration: {}
    
    ###############################################
    '''.format(i)

    test_index = i
    success = 0
    failure = 0
    for index, subset in enumerate(sets):
        if index != test_index: 
            for image in subset:
                im = cv.imread('image_' + str(image) + '.jpeg')
                segIm = hr.segment(im)
                palmPos, fingerPos, fingerLen = hr.extract(im, segIm)
                
                #Number of fingers feature
                numFingers = len(fingerPos)
            
                #Fingerlength feature
                fingerSum = 0
                for length in fingerLen:
                    fingerSum = fingerSum + length
                
                #Longest betweeen-finger X distance
                shortIndex = 0
                longIndex = 0
                n = 0
                for finger in fingerPos:
                    if finger[0] < fingerPos[shortIndex][0]:
                        shortIndex = n
                    if finger[0] > fingerPos[longIndex][0]:
                        longIndex = n
                    n += 1
                longestDist = hr.dist(fingerPos[shortIndex][0], fingerPos[longIndex][0], fingerPos[shortIndex][1], fingerPos[longIndex][1])
            
                sampleData = [numFingers,fingerSum,longestDist]
                samples.append(sampleData)
                labels.append(label_array[image])

    
    #make svm
    scaler = preprocessing.StandardScaler().fit(samples)
    model = svm.SVC()
    model.fit(scaler.transform(samples),labels)
            
    #run test on reserved subset
    for image in sets[test_index]:
        im = cv.imread('image_' + str(image) + '.jpeg')
        segIm = hr.segment(im)
        palmPos, fingerPos, fingerLen = hr.extract(im, segIm)
                
        #Normalized FingerLength Feature
        numFingers = (len(fingerPos) - scaler.mean_[0]) / scaler.scale_[0]
    
        #Normalized FingerSum Feature
        fingerSum = 0
        for length in fingerLen:
            fingerSum = fingerSum + length
        fingerSum = (fingerSum - scaler.mean_[1]) / scaler.scale_[1]

        #Normalized longest X distance between fingertips feature
        shortIndex = 0
        longIndex = 0
        n = 0
        for finger in fingerPos:
            if finger[0] < fingerPos[shortIndex][0]:
                shortIndex = n
            if finger[0] > fingerPos[longIndex][0]:
                longIndex = n
            n += 1

        longestDist = hr.dist(fingerPos[shortIndex][0], fingerPos[longIndex][0], fingerPos[shortIndex][1], fingerPos[longIndex][1])
        longestDist = (longestDist - scaler.scale_[2]) / scaler.scale_[2]
        sampleFeatures = np.array([numFingers, fingerSum, longestDist])
        if model.predict(sampleFeatures)[0] == label_array[image]:
            success += 1
        else:
            failure += 1
            
    accuracy_array.append(float(success)/float(success + failure))

#Calculate final accuracy and standard deviation
accuracy_average = sum(accuracy_array)/len(accuracy_array)
np_accuracy_array = np.array(accuracy_array)
std = np.std(np_accuracy_array)
print "Average accuracy:", accuracy_average
print "Standard Deviation:", std
