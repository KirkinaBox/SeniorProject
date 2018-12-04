#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 15:25:56 2018

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



from random import randint
import math
import numpy as np
from scipy.misc import derivative
from scipy import signal #scipy.signal.unit_impulse for Kronecker delta

fcLayer2Weights = []
fcLayer3Weights = []


#---Function for training the neural network---
def train(images, scores, epochs):
    
    #---Random filter weight initialization (0-255 is the range for HSV values)---
    
    '''
    brightFilter = [[randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                    [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                    [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                    [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                    [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)]]
    print("BrightFilter", brightFilter)
                
    dimFilter =    [[randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                    [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                    [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                    [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                    [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)]]
    print("DimFilter", dimFilter)

    normalFilter = [[randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                    [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                    [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                    [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)],
                    [randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)]]
    print("NormalFilter", normalFilter)
    '''
    brightFilter = [[randint(0, 255), randint(0, 255), randint(0, 255)],
                    [randint(0, 255), randint(0, 255), randint(0, 255)],
                    [randint(0, 255), randint(0, 255), randint(0, 255)]]
                    
    print("BrightFilter", brightFilter)
                
    dimFilter =    [[randint(0, 255), randint(0, 255), randint(0, 255)],
                    [randint(0, 255), randint(0, 255), randint(0, 255)],
                    [randint(0, 255), randint(0, 255), randint(0, 255)]]
                    
    print("DimFilter", dimFilter)

    normalFilter = [[randint(0, 255), randint(0, 255), randint(0, 255)],
                    [randint(0, 255), randint(0, 255), randint(0, 255)],
                    [randint(0, 255), randint(0, 255), randint(0, 255)]]
                    
    print("NormalFilter", normalFilter)

    features = [brightFilter, dimFilter, normalFilter]

    global fcLayer2Weights
    global fcLayer3Weights
    
    iteration = 0
    
    
    for a in range(0, epochs): #Number of times whole training set will go through network
        print("epoch ", a)
        for b in range(0, len(images)): #for each image in the training set
            if (b == 69):
                print("image", b)
        
        
        
            #---Convolution step---
            window = 3
            wX = 0
            wY = 0
            filteredImageList = []
    
            for filter in range(0, len(features)): #for each feature
                filteredImage = []
                wX = 0
                wY = 0
                while (wY+window-1 < images[b].size[1]):
                    filteredRow = []
                    while (wX+window-1 < images[b].size[0]):
                        windowTotal = 0
                        fiY = 0
                        for a in range(wY, wY+window):
                            if (fiY < window):
                                fiX = 0
                                for b in range(wX, wX+window):
                                    if (fiX < window):
                                        #if (images[b].getpixel((b, a))[2] == features[filter][fiY][fiX]):
                                        if (abs(images[b].getpixel((b, a))[2] - features[filter][fiY][fiX]) <= 30):
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
            #---end Convolution step---   
         
            
            
            
            
            
            #---Max-Pooling step---    
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
            #---end Max-Pooling step---   
            
            
            
            
            
            #-----------------------------
            #---Fully-connected layers----
            #-----------------------------
            
            
            
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
            bias2 = 0 #Need to change
            weightMatrix2 = []
            for i in range(0, layer2num):
                weightList2 = weights(len(flattened), 2, iteration, i)
                weightMatrix2.append(weightList2)
                a = np.dot(weightList2, flattened) + bias2
                if (a < 0): #ReLU
                    a = 0
                
                layer2.append(a)
                
            #layer2 = softmax(layer2)
            #---end fully-connected layer: flattened -> layer2 step---
                
            
            
            
            
            
            
            #---Fully connected layer from 1x10 matrix to 1x3 matrix---
            #Weighted connections pointing to each node in layer3
            #reference: http://neuralnetworksanddeeplearning.com/chap2.html
            
            #layer3num = 3
            layer3num = 2
            layer3 = []
            bias3 = 0
            weightMatrix3 = []
            for i in range(0, layer3num):
                weightList3 = weights(len(layer2), 3, iteration, i)
                weightMatrix3.append(weightList3)
                a = np.dot(weightList3, layer2) + bias3 #might need something other than a dot product
                if (a < 0): #ReLU
                    a = 0
                
                layer3.append(a)
            layer3 = softmax(layer3)
            #---end fully-connected layer: layer2 -> layer3 step--- 
            
            
            
            
            
            
            
            #---Defining target vector based on score from CSV file--- 
            # targetVector = [dim, normal, bright]
            targetVector = []
            if (scores[b] == '0\r'):
                #targetVector = [1, 0, 0] #dim
                targetVector = [1, 0]
            #if (scores[b] == '1\r'):
                #targetVector = [0, 1, 0] #normal
            if (scores[b] == '2\r'):
                #targetVector = [0, 0, 1] #bright
                targetVector = [0, 1]
            #---End target vector definition---
             
            
            
            
            
            
            
            
            #---Output layer error calculation---
            #reference: http://neuralnetworksanddeeplearning.com/chap2.html
            #outputErrorList = []
            #outputError = 0
            #for j in range(0, len(layer3)):
                
                
                # ***Note: This algorithm might need editing and has code commented out for future reference***
                
                #errorA = np.array(layer3) - np.array(targetVector)
                #a = layer3[j] #might be something other than a single number
                #def f(a):
                #    return a
                #errorB = np.array(derivative(f, 1.0)) #converting to np.array and using np.multiply: reference: https://stackoverflow.com/questions/40034993/how-to-get-element-wise-matrix-multiplication-hadamard-product-in-numpy
                #errorAB = np.multiply(errorA, errorB)
                #outputErrorList.append(errorAB)
                #subError = 0.5 * pow((np.array(targetVector) - np.array(layer3)), 2)
                
                
                #subError = 0.5 * pow((targetVector[j] - layer3[j]), 2)
                #outputError += subError
                
                
            #Cross-entropy calculation for output layer error
            #reference: https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
            #outputError = (-1/(b+1))*((targetVector[0]*math.log10(layer3[0])) + ((1-targetVector[0])*math.log10(1-layer3[0])) + (targetVector[1]*math.log10(layer3[1])) + ((1-targetVector[1])*math.log10(1-layer3[1])) + (targetVector[2]*math.log10(layer3[2])) + ((1-targetVector[2])*math.log10(1-layer3[2])))  
            
            outputError = (-1/(b+1))*((targetVector[0]*math.log10(layer3[0])) + ((1-targetVector[0])*math.log10(1-layer3[0])) + (targetVector[1]*math.log10(layer3[1])) + ((1-targetVector[1])*math.log10(1-layer3[1]))) 
            
            #print("output error", outputError)
            #---end output layer error calculation---
                
            
            
            
            
            
            
            #---Derivatives of output layer cross-entropy---
            #reference: https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
            outputErrorD1 = -1 * ((targetVector[0]*(1/layer3[0])) + (1-targetVector[0])*(1/(1-layer3[0])))
            
            outputErrorD2 = -1 * ((targetVector[1]*(1/layer3[1])) + (1-targetVector[1])*(1/(1-layer3[1])))
            
            #outputErrorD3 = -1 * ((targetVector[2]*(1/layer3[2])) + (1-targetVector[2])*(1/(1-layer3[2])))
            
            #dEdOut = [outputErrorD1, outputErrorD2, outputErrorD3]
            dEdOut = [outputErrorD1, outputErrorD2]
            #---Derivatives of output layer with respect to input---
            #reference: https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
            #outputInputD1 = (math.exp(layer3[0])*(math.exp(layer3[1])+math.exp(layer3[2])))/math.pow((math.exp(layer3[0])+math.exp(layer3[1])+math.exp(layer3[2])), 2)
            outputInputD1 = (math.exp(layer3[0])*math.exp(layer3[1]))/math.pow((math.exp(layer3[0])+math.exp(layer3[1])), 2)
            
            #outputInputD2 = (math.exp(layer3[1])*(math.exp(layer3[0])+math.exp(layer3[2])))/math.pow((math.exp(layer3[0])+math.exp(layer3[1])+math.exp(layer3[2])), 2)
            outputInputD2 = (math.exp(layer3[1])*math.exp(layer3[0]))/math.pow((math.exp(layer3[0])+math.exp(layer3[1])), 2)
            
            #outputInputD3 = (math.exp(layer3[2])*(math.exp(layer3[0])+math.exp(layer3[1])))/math.pow((math.exp(layer3[0])+math.exp(layer3[1])+math.exp(layer3[2])), 2)
            
            #dOoutdOin = [outputInputD1, outputInputD2, outputInputD3]
            dOoutdOin = [outputInputD1, outputInputD2]
            
            
            '''
            #Weight adjustment for layer2->layer3 weights
            for e in range(0, len(weightMatrix3)):
                #print(e)
                #for f in range(0, len(weightMatrix3[e])-1):
                for f in range(0, 2):
                    #print(f)
                    fcLayer3Weights[e][f] = dEdOut[f]*dOoutdOin[f]*layer2[e]
            '''
            
            
            
            
            
            
                
            #---Error backpropagation for layer2---
            #reference: http://neuralnetworksanddeeplearning.com/chap2.html
            
            
            # ***Note: This algorithm might need editing and has code commented out for future reference***
            
            #layer2errorList = []
            #for k in range(0, layer2num):
                #errorAa = np.array(weightMatrix3).transpose()
            #layer2errorList = np.array(weightMatrix3).transpose()
            
            layer2errorList = np.transpose(weightMatrix3)
            
                #errorAb = np.array(outputErrorList)
                #print(errorAa)
           # print(outputError)
                #errorA = np.multiply(errorAa, outputError) #either np.dot or np.multiply
                
            for e in range (0, len(layer2errorList)):
                
                #layer2errorList[e] = [i * outputError for i in layer2errorList[e]]
                
                for f in range(0, len(layer2errorList[e])):
                    layer2errorList[e][f] = layer2errorList[e][f] * outputError
                    
                #a = layer2[k]
                #errorB = np.array(derivative(a, 1.0))
                #errorAB = np.multiply(errorA, errorB)
                #layer2errorList.append(errorAB)
            
    
            
            #---end layer2 error backpropagation---  
                
            
            
            
            
            
                
            #---Error backpropagation for convolution filters---
            featuresErrorList = []
            for f in range(0, len(features)):
                singleFeatureErrorList = []
                for g in range(0, len(features[f])):
                    singleFeatureRow = []
                    for h in range(0, len(features[f][g])):
                        errorAa = np.array(weightMatrix2).transpose()
                        errorAb = np.array(layer2errorList)
                        errorA = np.dot(errorAa, errorAb)
                        
                        
                        # ***Note: This algorithm might need editing and has code commented out for future reference***
                        
                        #a = features[f][g][h]
                        #errorB = np.array(derivative(a, 1.0))
                        allNodes = [math.exp(i) for i in features[f][g]]
                        restExceptCurrent = [math.exp(i) for i in features[f][g] if i != features[f][g][h]]
                        errorB = (math.exp(features[f][g][h])*np.sum(restExceptCurrent))/np.sum(allNodes)
                        # ***Need to figure out derivative of softmax function***
                        #errorAB = np.multiply(errorA, errorB)
                        errorAB = [i*errorB for i in errorA]
                        #singleFeatureRow.append(errorAB)
                        
                        
                        singleFeatureRow.append(errorAB) #need to replace with errorAB
                    singleFeatureErrorList.append(singleFeatureRow)
                featuresErrorList.append(singleFeatureErrorList)
            #---end convolution filters' error backpropagation---
            
            
            
            
            
            
            
            #---Gradient descent for output layer---
            #reference: http://neuralnetworksanddeeplearning.com/chap2.html
            
            if (b == 4):
                for n in range(0, len(weightMatrix3)):
                    for o in range(0, len(weightMatrix3[n])):
                    
                    
                        # ***Note: This algorithm might need editing and has code commented out for future reference***
                    
                        #gradientA = np.dot(np.array(outputErrorList[n][o]), np.array(layer2[o]).transpose())
                    
                    
                        gradientA = (1/(b+1)) * outputError * np.array(layer2[o]).transpose()
                        fcLayer3Weights[n][o] = fcLayer3Weights[n][o] - gradientA
                print("fcLayer3Weights", fcLayer3Weights)    
                   
            #---end output layer gradient descent---
               
            
            
            
            
            
            
            
            #---Gradient descent for layer2---
            #reference: http://neuralnetworksanddeeplearning.com/chap2.html
            if (b == 69):
                for l in range(0, len(weightMatrix2)):
                    for m in range(0, len(weightMatrix2[l])):
                    
                    
                    # ***Note: This algorithm might need editing and has code commented out for future reference***
                    
                    #print("layer2", layer2errorList[l])
                    #print("flattened", flattened[m])
                    #gradientA = (1/(b+1)) * np.dot(np.array(layer2errorList[l][m]), np.array(flattened[m]).transpose())
                    #fcLayer2Weights[l][m] = fcLayer2Weights[l][m] - gradientA
                    
                    
                        fcLayer2Weights[l][m] = fcLayer2Weights[l][m] - ((1/(b+1)) * outputError * fcLayer2Weights[l][m])
                print("fcLayer2Weights", fcLayer2Weights)
                    
                    
            #---end layer2 gradient descent---
              
            
            
            
            
            
            
            
            #---Gradient descent for convolution filters---
            # ***Note: This algorithm might need editing and has code commented out for future reference***
            
            #updatedFilterWeights = []
            for p in range(0, len(features)):
                #singleFilter = []
                for q in range(0, len(features[p])):
                    #singleFilterRow = []
                    for r in range(0, len(features[p][q])):
                        features[p][q][r] = math.ceil(features[p][q][r] - np.sum(featuresErrorList[p][q])) #not sure if I should use np.sum or just single error value
                        if (features[p][q][r] < 0):
                            #features[p][q][r] = 0
                            features[p][q][r] = features[p][q][r]*(-1)
                            while (features[p][q][r] > 255):
                                features[p][q][r] = math.ceil(features[p][q][r]/10)
                            #features[p][q][r] = features[p][q][r]/10
                        if (features[p][q][r] > 255):
                            #features[p][q][r] = 255
                            while (features[p][q][r] > 255):
                                features[p][q][r] = math.ceil(features[p][q][r]/10)
            #---end convolution filters' gradient descent---
             
            
            
            
            
                
            #increase iteration variable for each image    
            iteration += 1
        #increase iteration variable for each epoch
        iteration += 1
        print("Features", features)
    
    #return learned weights for use in neural_network_classify.classify function
    learnedValues = [features, fcLayer2Weights, fcLayer3Weights]
    return learnedValues    
#---end of train function---      




#---Function for initializing random weights for first visit, or pointing to updated weight list for later visits
def weights(length, layer, iteration, index):
    global fcLayer2Weights
    global fcLayer3Weights
    w = []
    if (iteration == 0):
        w.extend(np.random.normal(0.0, 1.0, length)) #Gaussian distribution, might use randn instead
        if (layer == 2):
            fcLayer2Weights.append(w) 
        if (layer == 3):
            fcLayer3Weights.append(w)
    else:
        if (layer == 2):
            w.extend(fcLayer2Weights[index])
        if (layer == 3):
            w.extend(fcLayer3Weights[index])
    return w
#---end weights function---




#---Function for putting activations through softmax equation---
#reference: https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
def softmax(layer):
    smSubList = []
    for i in range(0, len(layer)):
        smSubList.append(math.exp(layer[i]))
    smList = []
    for j in range(0, len(layer)):
        sm = math.exp(layer[j])/np.sum(smSubList)
        smList.append(sm)
    return smList
#---end softmax function---

    
    
    
    
    
       
    
 
    
