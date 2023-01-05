import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist
import pickle

# Read-in, format and normalise dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data() #takes a while

x = X_train.shape[1] #pixel width of image should be 28
y = X_train.shape[2] #pixel height of image should be 28
X_train = X_train.reshape(X_train.shape[0],x*y).T / 255 #normalise
X_test = X_test.reshape(X_test.shape[0],x*y).T  / 255 #normalise

#functions for forward propagation#

def init_params(size):
    #the 10s here are because we have 10 identifiable characters
    #params in range [-0.5,0.5]
    W1 = np.random.rand(10,size) - 0.5
    b1 = np.random.rand(10,1) - 0.5
    W2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5
    return W1,b1,W2,b2

def ReLU(Z):
    #activation function for the nodes in our layer
    #Rectifed Linear Unit - easier than sigmoid or tanh
    #x if x>0 or 0 if x<=0 
    return np.maximum(Z,0)

def softmax(Z):
    #Compute softmax values for each set of values
    #https://en.wikipedia.org/wiki/Softmax_function
    exp = np.exp(Z - np.max(Z))
    return exp / exp.sum(axis=0)

def forward_propagation(X,W1,b1,W2,b2):
    #annotated expected array/vector sizes
    Z1 = W1.dot(X) + b1 #10, m
    A1 = ReLU(Z1) #10, m
    Z2 = W2.dot(A1) + b2 #10, m
    A2 = softmax(Z2) #10, m
    return Z1, A1, Z2, A2

#functions for back propagation#

def derivative_ReLU(Z):
    #See ReLU above
    #gradient = 0 if Z<=0, else 1
    #bool will be fine.
    return Z > 0

def one_hot(Y):
    #Identification - highest weight is most likely ID or class
    #return an 0 vector with 1 only in the position correspondind to the value in Y'''
    one_hot_Y = np.zeros((Y.max()+1,Y.size)) #10 possible identifications 0-9
    one_hot_Y[Y,np.arange(Y.size)] = 1 
    return one_hot_Y 

def backward_propagation(X, Y, A1, A2, W2, Z1, m):
    #again commented array/vector sizes
    one_hot_Y = one_hot(Y)
    dZ2 = 2*(A2 - one_hot_Y) #10,m
    dW2 = 1/m * (dZ2.dot(A1.T)) # 10 , 10
    db2 = 1/m * np.sum(dZ2,1) # 10, 1
    #weights transposed is to apply weights kind of in reverse to get the error of the first layer
    #using the derivative of our activation function
    dZ1 = W2.T.dot(dZ2)*derivative_ReLU(Z1) # 10, m
    dW1 = 1/m * (dZ1.dot(X.T)) #10, 784       (784 is 28^2)
    db1 = 1/m * np.sum(dZ1,1) # 10, 1

    return dW1, db1, dW2, db2

def update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2):
    #alpha is our learning rate
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(db1, (10,1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(db2, (10,1))

    return W1, b1, W2, b2

#Main functions#
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    #again bool is fine
    return np.sum(predictions == Y)/Y.size

def gradient_descent(X, Y, alpha, iterations):
    #learning rate alpha of 0.1-0.2, iterations >100 should be ok
    size , m = X.shape
    W1, b1, W2, b2 = init_params(size)

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, A1, A2, W2, Z1, m)

        W1, b1, W2, b2 = update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2)   
        #show accuracy every 10th iteration
        if (i+1) % int(iterations/10) == 0:
            print(f"Iteration: {i+1} / {iterations}")
            prediction = get_predictions(A2)
            print(f'{get_accuracy(prediction, Y):.2%}')
    return W1, b1, W2, b2

def make_predictions(X, W1 ,b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = get_predictions(A2)
    return predictions

def show_prediction(index,X, Y, W1, b1, W2, b2):
    #This will display the image and prediction for user verification
    
    # None means it'll make a new column
    # to meet dimension needs of make_predictions
    vect_X = X[:, index,None]
    prediction = make_predictions(vect_X, W1, b1, W2, b2)
    label = Y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = vect_X.reshape((x, y)) * 255

    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


####MAIN####
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.125, 300)
#store output params
with open("trained_params.pkl","wb") as param_dump:
    pickle.dump((W1, b1, W2, b2),param_dump)

#use stored params
with open("trained_params.pkl","rb") as param_dump:
    W1, b1, W2, b2=pickle.load(param_dump)
show_prediction(0,X_test, Y_test, W1, b1, W2, b2)
show_prediction(1,X_test, Y_test, W1, b1, W2, b2)
show_prediction(2,X_test, Y_test, W1, b1, W2, b2)
show_prediction(100,X_test, Y_test, W1, b1, W2, b2)
show_prediction(200,X_test, Y_test, W1, b1, W2, b2)


