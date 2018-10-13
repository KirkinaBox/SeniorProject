#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:23:20 2018

@author: jamiecrow
"""

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
    metaX = 0
    metaY = 0
    filteredImageList = []
    
    for filter in range (0, len(features)):
        filteredImage = []
        for a in range (metaY, testImage.size[1]-window):
            filteredRow = []
            for b in range (metaX, testImage.size[0]-window):
                #sliding the window down the row
                windowTotal = 0
                for c in range (wY, wY+window-1):
                    for d in range (wX, wX+window-1):
                        if (testImage.getpixel((d, c))[2] == filter[c][d]):
                            windowTotal += 1
                        else:
                            windowTotal += -1
                
                average = windowTotal/(Math.pow(window, 2))
                if (average < 0): #ReLU step
                    filteredRow.append(0)
                else:
                    filteredRow.append(average)   
                wX += 1
            filteredImage.append(filteredRow)
            wY += 1
        filteredImageList.append(filteredImage)
                
    return(self.pooling(filteredImageList)) #don't think this'll work without pooling() attached to a parent
    
    
def relu():
    #Rectified Linear Unit (ReLU) step goes here
    #swap negative numbers for 0
    #source: https://brohrer.github.io/how_convolutional_neural_networks_work.html
    
def pooling(self, listOfImages):
    #Pooling step goes here
    

    
def fc(testScore):
    #Fully Connected layer goes here (I think)
    #will probably use Softmax function
        #sources: https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
        #         http://cs231n.github.io/linear-classify/#softmax       
    
    # targetVector = [dim, normal, bright]
    if (testScore == 0):
        targetVector = [1, 0, 0] #dim
    if (testScore == 1):
        targetVector = [0, 1, 0] #normal
    if (testScore == 2):
        targetVector = [0, 0, 1] #bright
