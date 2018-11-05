#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 15:25:56 2018

@author: jamiecrow
"""

from PIL import Image
from random import randint
import math
import numpy as np
from scipy.misc import derivative

fcLayer2Weights = []
fcLayer3Weights = []


def train(images, scores, epochs):
    
    
    #need to figure out why each filter produces the same random number
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

    global fcLayer2Weights
    global fcLayer3Weights
    
    iteration = 0

    for a in range(0, epochs-1):
        print("epoch ", a)
        for b in range(0, len(images)):
        
            #Convolution
            window = 5
            wX = 0
            wY = 0
            filteredImageList = []
    
            for filter in range(0, len(features)): #for each feature
                filteredImage = []
                wX = 0
                wY = 0
                #print("width", images[b].size[0])
                #print("height", images[b].size[1])
                #while (wY+window-1 <= len(features[filter])):
                while (wY+window-1 < images[b].size[1]):
                    filteredRow = []
                    #while (wX+window-1 <= len(features[filter][0])):
                    while (wX+window-1 < images[b].size[0]):
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
                                        if (images[b].getpixel((b, a))[2] == features[filter][fiY][fiX]):
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
                
            print("filteredImagedList", filteredImageList)
            
            
            
            #Max-Pooling    
            stride = 2
            poolImages = []
            for e in range (0, len(filteredImageList)): #for each image
                poolSingle = []
                wX = 0
                wY = 0
                while (wY+stride < len(filteredImageList[e])):
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
                
            
            
            #print("poolImages", poolImages)
            #Fully-Connected Layers
            
            #classification = ""
    
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
            print("flattened", flattened)
            
            
            #Weighted connections pointing to each node in layer2
            #source: http://neuralnetworksanddeeplearning.com/chap2.html
            layer2num = 10
            layer2 = []
            bias2 = 0 #???
            weightMatrix2 = []
            for i in range(0, layer2num):
                weightList2 = weights(len(flattened), 2, iteration, i)
                weightMatrix2.append(weightList2)
                #weightMatrix.append(weightList)
                a = np.dot(weightList2, flattened) + bias2
                layer2.append(a)
            layer2 = softmax(layer2)
            print("layer2", layer2)
                
            
            #Weighted connections pointing to each node in layer3
            #source: http://neuralnetworksanddeeplearning.com/chap2.html
            #Might still need Softmax equation
            layer3num = 3
            layer3 = []
            bias3 = 0
            weightMatrix3 = []
            for i in range(0, layer3num):
                weightList3 = weights(len(layer2), 3, iteration, i)
                weightMatrix3.append(weightList3)
                a = np.dot(weightList3, layer2) + bias3 #might need something other than a dot product
                layer3.append(a)
            layer3 = softmax(layer3)
             
            
            #print(layer3)
            # targetVector = [dim, normal, bright]
            targetVector = []
            if (scores[b] == '0\r'):
                targetVector = [1, 0, 0] #dim
            if (scores[b] == '1\r'):
                targetVector = [0, 1, 0] #normal
            if (scores[b] == '2\r'):
                targetVector = [0, 0, 1] #bright
             
            #print(scores[b])
            #print(targetVector)
            #print(layer3)
            
            #Output layer error calculation
            #source: http://neuralnetworksanddeeplearning.com/chap2.html
            #outputErrorList = []
            outputError = 0
            for j in range(0, len(layer3)):
                #errorA = np.array(layer3) - np.array(targetVector)
                #a = layer3[j] #might be something other than a single number
                #def f(a):
                #    return a
                #errorB = np.array(derivative(f, 1.0)) #converting to np.array and using np.multiply: source: https://stackoverflow.com/questions/40034993/how-to-get-element-wise-matrix-multiplication-hadamard-product-in-numpy
                #errorAB = np.multiply(errorA, errorB)
                #outputErrorList.append(errorAB)
                #subError = 0.5 * pow((np.array(targetVector) - np.array(layer3)), 2)
                subError = 0.5 * pow((targetVector[j] - layer3[j]), 2)
                #print(layer3[j])
                outputError += subError
            
                
                
            #Error backpropagation for layer2
            #source: http://neuralnetworksanddeeplearning.com/chap2.html
            
            #layer2errorList = []
            #for k in range(0, layer2num):
                #errorAa = np.array(weightMatrix3).transpose()
            #layer2errorList = np.array(weightMatrix3).transpose()
            layer2errorList = np.transpose(weightMatrix3)
                #errorAb = np.array(outputErrorList)
                #print(errorAa)
            print(outputError)
                #errorA = np.multiply(errorAa, outputError) #either np.dot or np.multiply
            for e in range (0, len(layer2errorList)):
                #layer2errorList[e] = [i * outputError for i in layer2errorList[e]]
                for f in range(0, len(layer2errorList[e])):
                    layer2errorList[e][f] = layer2errorList[e][f] * outputError
                #a = layer2[k]
                #errorB = np.array(derivative(a, 1.0))
                #errorAB = np.multiply(errorA, errorB)
                #layer2errorList.append(errorAB)
             
                
                
            #Error backpropagation for convolution filters
            
            #global features
            featuresErrorList = []
            for f in range(0, len(features)):
                singleFeatureErrorList = []
                for g in range(0, len(features[f])):
                    singleFeatureRow = []
                    for h in range(0, len(features[f][g])):
                        errorAa = np.array(weightMatrix2).transpose()
                        errorAb = np.array(layer2errorList)
                        errorA = np.dot(errorAa, errorAb)
                       # a = features[f][g][h]
                        #errorB = np.array(derivative(a, 1.0))
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
             
                
                
            iteration += 1
        iteration += 1
        
    learnedValues = [features, fcLayer2Weights, fcLayer3Weights]
    return learnedValues          




#Function for initializing random weights for first visit, or pointing to updated weight list for later visits
def weights(length, layer, iteration, index):
    global fcLayer2Weights
    global fcLayer3Weights
    w = []
    if (iteration == 0):
        #print(np.random.random_sample(3))
        w.extend(np.random.normal(length)) #Gaussian distribution
        if (layer == 2):
            fcLayer2Weights.append(w) 
        if (layer == 3):
            fcLayer3Weights.append(w)
    else:
        if (layer == 2):
            w.extend(fcLayer2Weights[index])
        if (layer == 3):
            w.extend(fcLayer3Weights[index])
    #print(w)
    return w


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

#def convolution(self, testImage, testScore, iteration):
    
    #global features
    #stride = 1
   
    
                
    #return(self.pooling(filteredImageList, testScore, iteration)) #don't think this'll work without pooling() attached to a parent
    
    
#def relu():
    #Rectified Linear Unit (ReLU) step goes here
    #swap negative numbers for 0
    #source: https://brohrer.github.io/how_convolutional_neural_networks_work.html
    
#def pooling(self, listOfImages, testScore, iteration):
    
     
    #return(self.fc(poolImages, testScore, iteration))



            
        
#def fc(self, poolImages, testScore, iteration):
    #Fully Connected layer goes here (I think)
    
    #will probably use Softmax function
        #sources: https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
        #         http://cs231n.github.io/linear-classify/#softmax   
        #         https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
    
    #Flattening cube (poolImages) into vector
    #source: https://www.quora.com/What-is-the-meaning-of-flattening-step-in-a-convolutional-neural-network
    #        answer by Alex Coninx
    
    
    
    
    
       
    
 
    
