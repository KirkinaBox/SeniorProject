#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 23:10:39 2018

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
from neural_network_epochs import train
from neural_network_classify import classify
import numpy as np
from matplotlib import pyplot
from matplotlib import animation
import mimetypes
from os import path
import pickle


def userInput():

    print("Input an image file path: ")
    testImage = ""
    
    
    #Error handling to check for valid file type and valid file name
    while (testImage == ""):
        try:
            filePath = raw_input()
            filePath = imageCheck(filePath)
            testImage = Image.open(filePath)
        
        except IOError:
            print("This file cannot be found. Please input a valid file name: ")
        
    
    testImage = testImage.resize((testImage.size[0]/140, testImage.size[1]/140))
    testImage = testImage.convert("HSV")
    
    pickle_off = open("cnn.pickle", "rb")
    learnedValues = pickle.load(pickle_off)
    features = learnedValues[0]
    weights2 = learnedValues[1]
    weights3 = learnedValues[2]
    
    classification = classify(testImage, features, weights2, weights3)
    print(classification)
    eye(classification)
    
    print("Enter to continue, or type 'exit' to stop: ")
    if (raw_input() == "exit"):
        return
    else:
        userInput()
    


#Function to generate visual output
def eye(classification):
    #defaultSize = 0.75
    
    pupilSize = 0
    if (classification == "dim"):
        pupilSize = 1.0
    if (classification == "normal"):
        pupilSize = 0.75
    if (classification == "bright"):
        pupilSize = 0.5
    #pupil = 0.75
    
    
    
    figure = pyplot.figure()
    figure.set_dpi(300)
    figure.set_size_inches(1, 1)
    
    pupil = pyplot.Circle((0, 0), color="black", radius=pupilSize)
    iris = pyplot.Circle((0, 0), radius=1.5, color="saddlebrown")
    pyplot.gca().add_patch(iris)
    pyplot.gca().add_patch(pupil)
    pyplot.axis("scaled")
    pyplot.axis("off")
    
    
    '''
    def init():
         pupil.set_radius(defaultSize)
         return pupil
    
    def animate(i):
        radius = pupil.radius
        #endpoint = 0
        if (classification == "dim"):
            #endpoint = 1.0
            radius = defaultSize + (0.01*i)
        if (classification == "normal"):
            #endpoint = 0.75
            radius = defaultSize
        if (classification == "bright"):
            #endpoint = 0.5
            radius = defaultSize - (0.01*i)
        #pupil.radius = radius
        pupil.set_radius(radius)
        print(pupil.radius)
        return pupil
        
    anim = animation.FuncAnimation(figure, animate, init_func=init, frames=25)
    '''
    
    
    
    pyplot.show()



#Function to check if input is a jpeg file    
def imageCheck(filePath):
    fileType = mimetypes.guess_type(filePath)
    returnFile = filePath
    #print(fileType[0])
    if (fileType[0] != "image/jpeg"):
        print("Invalid file type. Please input a file path ending in .jpeg or .jpg")
        newFilePath = raw_input()
        returnFile = imageCheck(newFilePath)
    
    
    return returnFile
    



userInput()