#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:42:21 2018

@author: jamiecrow
"""

from PIL import Image
from random import randint
import math
import numpy as np
from scipy.misc import derivative

def classify(image, features, weights2, weights3):
    
    #Convolution
   # window = 5
    #wX = 0
    #wY = 0
    #filteredImageList = []
    
    #for filter in range (0, len(features)): #for each feature
        #filteredImage = []
        #wX = 0
        #wY = 0
        #while (wY+window-1 <= len(features[filter])):
            #filteredRow = []
            #while (wX+window-1 <= len(features[filter][0])):
                #windowTotal = 0
                #for a in range (wY, wY+window-1):
                    #for b in range (wX, wX+window-1):
                        #if (image.getpixel((wX, wY))[2] == filter[wY][wX]):
                            #windowTotal += 1
                        #else:
                            #windowTotal += -1
                #average = windowTotal/(math.pow(window, 2))
                #if (average < 0): #ReLU step
                    #filteredRow.append(0)
                #else:
                    #filteredRow.append(average)
                #wX += 1
            #filteredImage.append(filteredRow)
            #wY += 1
        #filteredImageList.append(filteredImage)
        
        
    window = 5
    wX = 0
    wY = 0
    filteredImageList = []
    
    for filter in range(0, len(features)): #for each feature
        filteredImage = []
        wX = 0
        wY = 0
        #print("width", image.size[0])
        #print("height", image.size[1])
        #while (wY+window-1 <= len(features[filter])):
        while (wY+window-1 < image.size[1]):
            filteredRow = []
            #while (wX+window-1 <= len(features[filter][0])):
            while (wX+window-1 < image.size[0]):
                windowTotal = 0
                fiY = 0
                for a in range(wY, wY+window):
                    if (fiY < 5):
                        fiX = 0
                        for b in range(wX, wX+window):
                            if (fiX < 5):
                                #print(wX, wY, a, b)
                                #if((a == 0) and (b == 5)):
                                #print(features[filter][wY][wX])
                                if (image.getpixel((b, a))[2] == features[filter][fiY][fiX]):
                                    windowTotal += 1
                                else:
                                    windowTotal += -1
                                fiX += 1
                        fiY += 1
                average = windowTotal/(pow(window, 2))
                if (average < 0): #ReLU step
                    filteredRow.append(0)
                else:
                    filteredRow.append(average)
                wX += 1
            filteredImage.append(filteredRow)
            wX = 0
            wY += 1
        filteredImageList.append(filteredImage)
                
             
                
    #Max-Pooling    
    stride = 2
    poolImages = []
    for e in range (0, len(filteredImageList)): #for each image
        poolSingle = []
        wX = 0
        wY = 0
        while (wY+stride <= len(filteredImageList[e])):
            poolSingleRow=[]
            while (wX+stride <= len(filteredImageList[e][0])):
                uno = filteredImageList[e][wY][wX]
                dos = filteredImageList[e][wY][wX+1]
                tres = filteredImageList[e][wY+1][wX]
                cuatro = filteredImageList[e][wY+1][wX+1]
                windowMax = max(uno, dos, tres, cuatro)
                poolSingleRow.append(windowMax)
                wX = wX+stride
            poolSingle.append(poolSingleRow)
            wX = 0
            wY = wY+stride
        poolImages.append(poolSingle)
            
            
            
            
    #Fully-connected layers
    classification = ""
    
    
    #softmaxSum = 0
    flattened = []
    for l in range (0, len(poolImages)):
        for m in range (0, len(poolImages[l])):
            for n in range (0, len(poolImages[l][m])):
                #don't think Softmax should be in this step
                #need to figure that out
                
                #node = math.exp(poolImages[l][m][n]) #Softmax
                #softmaxSum += node #Softmax
                #flattened.append(node)
                flattened.append(poolImages[l][m][n])
                    
                    
    #Weighted connections pointing to each node in layer2
    #source: http://neuralnetworksanddeeplearning.com/chap2.html
    layer2num = 10
    layer2 = []
    bias2 = 0 #???
    #weightMatrix2 = []
    for i in range(0, layer2num):
        #weightList2 = self.weights(len(flattened), 2, iteration, i)
        #weightMatrix2.append(weightList2)
        #weightMatrix.append(weightList)
        a = np.dot(weights2[i], flattened) + bias2
        layer2.append(a)
    layer2 = softmax(layer2)
            
            
    #Weighted connections pointing to each node in layer3
    #source: http://neuralnetworksanddeeplearning.com/chap2.html
    layer3num = 3
    layer3 = []
    bias3 = 0
    #weightMatrix3 = []
    for i in range(0, layer3num):
        #weightList3 = self.weights(layer2num, 3, iteration, i)
        #weightMatrix3.append(weightList3)
        a = np.dot(weights3[i], layer2) + bias3 #might need something other than a dot product
        layer3.append(a)
    layer3 = softmax(layer3)
            
            
    if ((layer3[0] > layer3[1]) and (layer3[0] > layer3[2])):
        classification = "dim"
    if ((layer3[1] > layer3[0]) and (layer3[1] > layer3[2])):
        classification = "normal"
    if ((layer3[2] > layer3[0]) and (layer3[2] > layer3[1])):
        classification = "bright"
            
            
    return classification




#Function for putting activations through softmax equation
#source: https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
def softmax(layer):
    smSubList = []
    for i in range(0, len(layer)):
        smSubList.append(math.exp(layer[i]))
    smList = []
    for j in range(0, len(layer)):
        sm = math.exp(layer[j])/np.sum(smSubList)
        smList.append(sm)
    return smList
    