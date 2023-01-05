# BabyNeuralNetwork
Making a simple NN to better understand some of tensorflow stuff.

This will create a NN to identify the mnist handwriting dataset
This dataset is a bunch of 28x28 px images of 0-9 character 
handwriting with labels in the training set.

Included is a set for training and a set for testing models. I think keras has the dataset too ("from keras.datasets import mnist", will use that to make readin easier).

Each image has has greyscale pixel colour in range 0-255 per pisel.

Can use pandas for data reading and numpy+matplotlib for the rest.

Notes:
Randomise data order before starting
Randomise starting factors to 0/pm0.5
Will need to mormalise all image values so max=1 instead of max=255

