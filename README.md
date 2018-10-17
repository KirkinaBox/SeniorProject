# Simulating Pupillary Light Reflex with a Neural Network

In Progress:
- Finishing implementation of fully-connected layer
  - Researching and implementing error calculation and Softmax function (and any other necessary calculations)
- Testing flow of neural network functions
- Testing flow of training images through neural network

Completed: 
 - 10/14/2018: Shortened and made corrections to convolution and pooling functions; implemented flattening in fully-connected layer
 - 10/13/2018: Made progress on pooling function
 - 10/12/2018: Started writing functions for convolution, pooling, and fully-connected layer; initialized filter matrices (one filter for each of the three image categories; defined target vectors based on training image scores (for use in fully-connected layer); implemented convolution function
 - 10/9/2018: Implemented training image resizing (shrinking) and converting color mode from RGB to HSV
 - 10/8/2018: Started a Python file for reading in and preprocessing the training images
 - 10/7/2018: Created a csv file with two columns, one with file paths for each training image and the other with the image's corresponding score. "dim" has a score of 0, "normal" has a score of 1, and "bright" has a score of 2
 - 9/28/2018: Created a set of 105 images for training the neural network, with each category of images (normal, bright, and dim) making up roughly one-third of the total image set
