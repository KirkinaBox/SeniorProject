#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 23:41:55 2018

@author: jamiecrow
"""

from PIL import Image
from random import randint

file = open("TrainingImages_withScores.csv", "r")
trainingList = file.read()
trainingList = trainingList.split("\n")
#print(trainingList)

for i in range (1, len(trainingList)):
    trainingList[i] = trainingList[i].split(",")
    #print(trainingList[i][0])
    image = Image.open(trainingList[i][0])
    image = image.resize((image.size[0]/100, image.size[1]/100))
    image = image.convert("HSV")
    print("converted")
    score = trainingList[i][1]
  

image.show()
#print(image.getpixel((14, 79))[2])
print(image.size[0])
print(image.size[1])
#print(randint(0, 255))


#---------------------------------------------------------------------------------------------------------------------------------------



    
    
        
    