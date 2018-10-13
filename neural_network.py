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

stride = 2

def convolution(testImage, testScore, features):
    
    
    # targetVector = [dim, normal, bright]
    #if (testScore == 0):
        #targetVector = [1, 0, 0] #dim
    #if (testScore == 1):
        #targetVector = [0, 1, 0] #normal
    #if (testScore == 2):
        #targetVector = [0, 0, 1] #bright
    
    
def relu():
    #Rectified Linear Unit (ReLU) step goes here
    #swap negative numbers for 0
    #source: https://brohrer.github.io/how_convolutional_neural_networks_work.html
    
def pooling():
    #Pooling step goes here
   

    
def fc(testScore):
    #Fully Connected layer goes here (I think)
    #will probably use Softmax function
        #sources: https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
        #         http://cs231n.github.io/linear-classify/#softmax       

    if (testScore == 0):
        targetVector = [1, 0, 0] #dim
    if (testScore == 1):
        targetVector = [0, 1, 0] #normal
    if (testScore == 2):
        targetVector = [0, 0, 1] #bright
