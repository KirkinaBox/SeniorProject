#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:23:20 2018

@author: jamiecrow
"""
from PIL import Image
from random import randint
import math


brightFilter = [[randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)]]
                
dimFilter =    [[randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)]]

normalFilter = [[randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)]]

featureMap = [brightFilter, dimFilter, normalFilter]



def convolution(self, testImage, testScore, features):
    
    #stride = 1
    window = 5
    wX = 0
    wY = 0
    filteredImageList = []
    
    for filter in range (0, len(features)): #for each feature
        filteredImage = []
        wX = 0
        wY = 0
        while (wY+window-1 <= len(features[filter])):
            filteredRow = []
            while (wX+window-1 <= len(features[filter][0])):
                windowTotal = 0
                for a in range (wY, wY+window-1):
                    for b in range (wX, wX+window-1):
                        if (testImage.getpixel((wX, wY))[2] == filter[wY][wX]):
                            windowTotal += 1
                        else:
                            windowTotal += -1
                average = windowTotal/(math.pow(window, 2))
                if (average < 0): #ReLU step
                    filteredRow.append(0)
                else:
                    filteredRow.append(average)
                wX += 1
            filteredImage.append(filteredRow)
            wY += 1
        filteredImageList.append(filteredImage)
    
                
    return(self.pooling(filteredImageList, testScore)) #don't think this'll work without pooling() attached to a parent
    
    
#def relu():
    #Rectified Linear Unit (ReLU) step goes here
    #swap negative numbers for 0
    #source: https://brohrer.github.io/how_convolutional_neural_networks_work.html
    
def pooling(self, listOfImages, testScore):
    #Pooling step goes here
    #window = 2
    stride = 2
    
    poolImages = []
    for e in range (0, len(listOfImages)): #for each image
        poolSingle = []
        wX = 0
        wY = 0
        while (wY+stride <= len(listOfImages[e])):
            poolSingleRow=[]
            while (wX+stride <= len(listOfImages[e][0])):
                uno = listOfImages[e][wY][wX]
                dos = listOfImages[e][wY][wX+1]
                tres = listOfImages[e][wY+1][wX]
                cuatro = listOfImages[e][wY+1][wX+1]
                windowMax = max(uno, dos, tres, cuatro)
                poolSingleRow.append(windowMax)
                wX = wX+stride
            poolSingle.append(poolSingleRow)
            wX = 0
            wY = wY+stride
        poolImages.append(poolSingle)
     
    return(self.fc(poolImages, testScore))
            
        
def fc(self, poolImages, testScore):
    #Fully Connected layer goes here (I think)
    #will probably use Softmax function
        #sources: https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
        #         http://cs231n.github.io/linear-classify/#softmax       
    
    #Flattening cube (poolImages) into vector
    flattened = []
    for l in range (0, len(poolImages)):
        for m in range (0, len(poolImages[l])):
            for n in range (0, len(poolImages[l][m])):
                flattened.append(poolImages[l][m][n])
    
    
    # targetVector = [dim, normal, bright]
    if (testScore == 0):
        targetVector = [1, 0, 0] #dim
    if (testScore == 1):
        targetVector = [0, 1, 0] #normal
    if (testScore == 2):
        targetVector = [0, 0, 1] #bright
    
    
    #Error calculation
    #source: https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/