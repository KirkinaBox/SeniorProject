#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:42:21 2018

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
from random import randint
import math
import numpy as np
from scipy.misc import derivative


#---Function for running an input image through the neural network and returning a classification---
def classify(image, features, weights2, weights3):

    #---Convolution step---
    window = 5
    wX = 0
    wY = 0
    filteredImageList = []
    
    for filter in range(0, len(features)): #for each feature
        filteredImage = []
        wX = 0
        wY = 0
        while (wY+window-1 < image.size[1]):
            filteredRow = []
            while (wX+window-1 < image.size[0]):
                windowTotal = 0
                fiY = 0
                for a in range(wY, wY+window):
                    if (fiY < 5):
                        fiX = 0
                        for b in range(wX, wX+window):
                            if (fiX < 5):
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
    #---End convolution step---
                
                        
    #---Max-Pooling step---  
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
    #---end max-pooling step---
            
            
    #---Fully-connected layers---
    #----------------------------
    
    #---Flattening max-pooled 3D matrix into 1D matrix---
    flattened = []
    for l in range (0, len(poolImages)):
        for m in range (0, len(poolImages[l])):
            for n in range (0, len(poolImages[l][m])):
                flattened.append(poolImages[l][m][n])
    #---end flattening step---
                    
    
    #---Fully-connected layer from flattened 1D matrix to 1x10 matrix---                
    #Weighted connections pointing to each node in layer2
    #reference: http://neuralnetworksanddeeplearning.com/chap2.html
    layer2num = 10
    layer2 = []
    bias2 = 0 #???
    for i in range(0, layer2num):
        a = np.dot(weights2[i], flattened) + bias2
        layer2.append(a)
    layer2 = softmax(layer2)
    #---end fully-connected layer: flattened -> layer2 step---
            
     
    #---Fully connected layer from 1x10 matrix to 1x3 matrix---       
    #Weighted connections pointing to each node in layer3
    #reference: http://neuralnetworksanddeeplearning.com/chap2.html
    layer3num = 3
    layer3 = []
    bias3 = 0
    for i in range(0, layer3num):
        a = np.dot(weights3[i], layer2) + bias3 #might need something other than a dot product
        layer3.append(a)
    layer3 = softmax(layer3)
    #---end fully-connected layer: layer2 -> layer3 step---
    
    print("OutputLayer", layer3)
    #---Classification step---   
    classification = ""      
    if ((layer3[0] > layer3[1]) and (layer3[0] > layer3[2])):
        classification = "dim"
    elif ((layer3[1] >= layer3[0]) and (layer3[1] >= layer3[2])):
        classification = "normal"
    elif ((layer3[2] > layer3[0]) and (layer3[2] > layer3[1])):
        classification = "bright"
    else:
        classification = "normal1"
    
    
    #---end classification step---
            
            
    return classification
#---end of classify function---




#---Function for putting activations through softmax equation---
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
#---end of softmax function---
    