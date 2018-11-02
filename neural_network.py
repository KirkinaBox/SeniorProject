#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:23:20 2018

@author: jamiecrow
"""
from PIL import Image
from random import randint
import math
import numpy as np
from scipy.misc import derivative


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

features = [brightFilter, dimFilter, normalFilter]

fcLayer2Weights = []
fcLayer3Weights = []


def convolution(self, testImage, testScore, iteration):
    
    global features
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
    
                
    return(self.pooling(filteredImageList, testScore, iteration)) #don't think this'll work without pooling() attached to a parent
    
    
#def relu():
    #Rectified Linear Unit (ReLU) step goes here
    #swap negative numbers for 0
    #source: https://brohrer.github.io/how_convolutional_neural_networks_work.html
    
def pooling(self, listOfImages, testScore, iteration):
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
     
    return(self.fc(poolImages, testScore, iteration))



            
        
def fc(self, poolImages, testScore, iteration):
    #Fully Connected layer goes here (I think)
    
    #will probably use Softmax function
        #sources: https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
        #         http://cs231n.github.io/linear-classify/#softmax   
        #         https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
    
    #Flattening cube (poolImages) into vector
    #source: https://www.quora.com/What-is-the-meaning-of-flattening-step-in-a-convolutional-neural-network
    #        answer by Alex Coninx
    
    global fcLayer2Weights
    global fcLayer3Weights
    
    
    # targetVector = [dim, normal, bright]
    targetVector = []
    if (testScore == 0):
        targetVector = [1, 0, 0] #dim
    if (testScore == 1):
        targetVector = [0, 1, 0] #normal
    if (testScore == 2):
        targetVector = [0, 0, 1] #bright
    
    
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
    weightMatrix2 = []
    for i in range(0, layer2num-1):
        weightList2 = self.weights(len(flattened), 2, iteration)
        weightMatrix2.append(weightList2)
        #weightMatrix.append(weightList)
        a = np.dot(weightList2, flattened) + bias2
        layer2.append(a)
     
    
    #Weighted connections pointing to each node in layer3
    #source: http://neuralnetworksanddeeplearning.com/chap2.html
    layer3num = 3
    layer3 = []
    bias3 = 0
    weightMatrix3 = []
    for i in range(0, layer3num-1):
        weightList3 = self.weights(layer2num, 3, iteration)
        weightMatrix3.append(weightList3)
        a = np.dot(weightList3, layer2) + bias3 #might need something other than a dot product
        layer3.append(a)
     
        
    #Output layer error calculation
    #source: http://neuralnetworksanddeeplearning.com/chap2.html
    outputErrorList = []
    for j in range(0, layer3num-1):
        errorA = np.array(layer3 - targetVector)
        a = layer3[j]
        errorB = np.array(derivative(a, 1.0)) #converting to np.array and using np.multiply: source: https://stackoverflow.com/questions/40034993/how-to-get-element-wise-matrix-multiplication-hadamard-product-in-numpy
        errorAB = np.multiply(errorA, errorB)
        outputErrorList.append(errorAB)
   
    
    #Error backpropagation for layer2
    #source: http://neuralnetworksanddeeplearning.com/chap2.html
    layer2errorList = []
    for k in range(0, layer2num-1):
        errorAa = np.array(weightMatrix3).transpose()
        errorAb = np.array(outputErrorList)
        errorA = np.dot(errorAa, errorAb) #either np.dot or np.multiply
        a = layer2[k]
        errorB = np.array(derivative(a, 1.0))
        errorAB = np.multiply(errorA, errorB)
        layer2errorList.append(errorAB)
    
    
    #Error backpropagation for convolution filters
    global features
    featuresErrorList = []
    for f in range(0, len(features)):
        singleFeatureErrorList = []
        for g in range(0, len(features[f])):
            singleFeatureRow = []
            for h in range(0, len(features[f][g])):
                errorAa = np.array(weightMatrix2).transpose()
                errorAb = np.array(layer2errorList)
                errorA = np.dot(errorAa, errorAb)
                a = features[f][g][h]
                errorB = np.array(derivative(a, 1.0))
                errorAB = np.multiply(errorA, errorB)
                singleFeatureRow.append(errorAB)
            singleFeatureErrorList.append(singleFeatureRow)
        featuresErrorList.append(singleFeatureErrorList)
    
    
    #Gradient descent for output layer
    #source: http://neuralnetworksanddeeplearning.com/chap2.html
    for n in range(0, len(weightMatrix3)):
        for o in range(0, len(weightMatrix3[n])):
            gradientA = np.dot(np.array(outputErrorList[n][o]), np.array(layer2[o]).transpose())
            fcLayer3Weights[n][o] = fcLayer3Weights[n][o] - gradientA   
        
    
    #Gradient descent for layer2
    #source: http://neuralnetworksanddeeplearning.com/chap2.html
    for l in range(0, len(weightMatrix2)):
        for m in range(0, len(weightMatrix2[l])):
            gradientA = np.dot(np.array(layer2errorList[l][m]), np.array(flattened[m]).transpose())
            fcLayer2Weights[l][m] = fcLayer2Weights[l][m] - gradientA
        
     
    #Gradient descent for convolution filters
    #updatedFilterWeights = []
    for p in range(0, len(features)):
        #singleFilter = []
        for q in range(0, len(features[p])):
            #singleFilterRow = []
            for r in range(0, len(features[p][q])):
                features[p][q][r] = features[p][q][r] - np.sum(featuresErrorList[p][q]) #not sure if I should use np.sum or just single error value
    
    
    
    #for s in range (0, len(flattened)):
        #flattened[s] = flattened[s]/softmaxSum #Softmax

    
    
     
    
    
def weights(self, length, layer, iteration):
    global fcLayer2Weights
    global fcLayer3Weights
    w = []
    if (iteration == 0):
        w = np.random.randn(length)
        if (layer == 2):
            fcLayer2Weights.append(w) 
        if (layer == 3):
            fcLayer3Weights.append(w)
        
    return w