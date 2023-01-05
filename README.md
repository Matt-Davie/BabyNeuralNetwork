# BabyNeuralNetwork
Making a simple NN to better understand some of tensorflow stuff.

This will create a NN to identify the mnist handwriting dataset
This dataset is a bunch of 28x28 px images of 0-9 characters 
handwritten, with labels in the training set.

Included is a set for training and a set for testing models if you don't want to mess with keras because you are a purist.
Will use "from keras.datasets import mnist", to make readin easier).

Each image has has greyscale pixel colour in range 0-255 per pixel.

Can use pickle for data reading and numpy+matplotlib for the rest.

Notes:
Randomise starting factors to 0/pm0.5
Will need to mormalise all image values so max=1 instead of max=255

#To do: explanatory maths and layer/node flow, other functions: tanh, sigmoid etc with derivatives.
#randomise data order - might make a difference in training?
