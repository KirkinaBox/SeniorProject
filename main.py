#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 23:41:55 2018

@author: jamiecrow
"""
'''
Author:   Jamie Crow
Sponsor:  Dr. Toshikazu Ikuta
          Digital Neuroscience Laboratory
          University of Mississippi Department of Communication Sciences & Disorders
Semester: Fall 2018
Class:    CSCI 487 (Senior Project)

Objective: 
    To simulate pupillary light reflex using a convolutional neural network written from scratch                             (i.e. without the use of Keras, OpenCV, Tensorflow, etc.). The program should accept a JPEG file as user    input, and should produce visual output in the form of a circle, representing the pupil, growing or shrinking in size in response to the output of the neural network. The neural network should classify images as either dim, normal, or bright.  
'''



from PIL import Image
from neural_network_epochs import train
from neural_network_classify import classify


#Reading in CSV file with image file paths in one column and scores in a second column
file = open("TrainingImages_withScores.csv", "r")
trainingList = file.read()
trainingList = trainingList.split("\n")


#Processing CSV file by opening each file path, shrinking each image, converting the color mode of each image from RGB to HSV, and adding each image and each score to a corresponding list
imageList = []
scoreList = []
for i in range (1, len(trainingList)):
    trainingList[i] = trainingList[i].split(",")
    image = Image.open(trainingList[i][0])
    image = image.resize((image.size[0]/140, image.size[1]/140))
    image = image.convert("HSV")
    imageList.append(image)
    print("converted", i)
    score = trainingList[i][1]
    scoreList.append(score)

   
#Train neural network using train function in neural_network_epochs.py
learnedValues = train(imageList, scoreList, 3)
features = learnedValues[0]
weights2 = learnedValues[1]
weights3 = learnedValues[2]


#User input and preprocessing for testing images
print("Input an image file path: ")
testImage = Image.open(raw_input())
testImage = testImage.resize((testImage.size[0]/140, testImage.size[1]/140))
testImage = testImage.convert("HSV")


#Run testing image through neural network in neural_network_classify.py
classification = classify(testImage, features, weights2, weights3)


#Print return value from classify function
#Will implement visual output later
print(classification)





    
    
        
    