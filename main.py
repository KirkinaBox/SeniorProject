#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 23:41:55 2018

@author: jamiecrow
"""

from PIL import Image
from random import randint
import math
from neural_network_epochs import train
from neural_network_classify import classify

file = open("TrainingImages_withScores.csv", "r")
trainingList = file.read()
trainingList = trainingList.split("\n")
#print(trainingList)

imageList = []
scoreList = []
for i in range (1, len(trainingList)):
    trainingList[i] = trainingList[i].split(",")
    #print(trainingList[i][0])
    image = Image.open(trainingList[i][0])
    image = image.resize((image.size[0]/140, image.size[1]/140))
    image = image.convert("HSV")
    imageList.append(image)
    print("converted", i)
    score = trainingList[i][1]
    scoreList.append(score)

#imageList[0].show()
#print(imageList[0].getpixel((0, 0))[2])
    
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


#image.show()
#print(image.getpixel((14, 79))[2])
#print(image.size[0])
#print(image.size[1])
#print(math.pow(2,3))
#print(randint(0, 255))

#print(scoreList[1])


#---------------------------------------------------------------------------------------------------------------------------------------



    
    
        
    