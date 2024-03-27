import sys
global verbose_output

def init_params(size):
    '''
    The 10s here are because we have 10 identifiable characters
    params in range [-0.5, 0.5]
    '''
    W1 = np.random.rand(10, size) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def reLU(Z):
    '''
    Activation function for the nodes in our layer
    Rectifed Linear Unit - easier than sigmoid or tanh
    x if x>0 or 0 if x<=0
    '''
    return np.maximum(Z, 0)

def softmax(Z):
    '''
    Compute softmax values for each set of values
    https://en.wikipedia.org/wiki/Softmax_function
    '''
    exp = np.exp(Z - np.max(Z))
    return exp / exp.sum(axis = 0)

def forward_propagation(X, W1, b1, W2, b2):
    '''
    Annotated expected array/vector sizes
    '''
    Z1 = W1.dot(X) + b1 # 10, m
    A1 = reLU(Z1) # 10, m
    Z2 = W2.dot(A1) + b2 # 10, m
    A2 = softmax(Z2) # 10, m

    return Z1, A1, Z2, A2

# Functions for back propagation#
def derivative_ReLU(Z):
    '''
    See ReLU above
    gradient = 0 if Z<=0, else 1
    Bool will be fine.
    '''
    return Z > 0

def one_hot(Y):
    '''
    Identification - highest weight is most likely ID or class
    Return an 0 vector with 1 only in the position correspondind to the value in Y'''
    ''''''
    one_hot_Y = np.zeros((Y.max() + 1, Y.size)) # 10 possible identifications 0-9
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def backward_propagation(X, Y, A1, A2, W2, Z1, m):
    # Commented array/vector sizes
    one_hot_Y = one_hot(Y)
    dZ2 = 2 * (A2 - one_hot_Y) # 10, m
    dW2 = 1 / m * (dZ2.dot(A1.T)) # 10, 10
    db2 = 1 / m * np.sum(dZ2, 1) # 10, 1

    # Weights transposed is to apply weights kind of in reverse to get the error of the first layer
    # using the derivative of our activation function
    dZ1 = W2.T.dot(dZ2) * derivative_ReLU(Z1) # 10, m
    dW1 = 1/m * (dZ1.dot(X.T)) # 10, 784
    db1 = 1/m * np.sum(dZ1, 1) # 10, 1

    return dW1, db1, dW2, db2

def update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2):
    '''
    Alpha is our learning rate
    '''
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(db1, (10, 1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(db2, (10, 1))

    return W1, b1, W2, b2

# Main functions #
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    '''
    learning rate alpha of 0.1-0.2, iterations >100 should be ok
    '''
    global verbose_output
    size, m = X.shape
    W1, b1, W2, b2 = init_params(size)

    final_acc = 0.0
    loader = "."
    display_chunks = 10

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, A1, A2, W2, Z1, m)

        W1, b1, W2, b2 = update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2)

        # Show accuracy every 10th iteration
        if (i + 1) % int(iterations / display_chunks) == 0:
            prediction = get_predictions(A2)
            final_acc = get_accuracy(prediction, Y)
            if verbose_output:
                print(f"Iteration: {i + 1} / {iterations}")
                print(f"Accuracy: {get_accuracy(prediction, Y):.2%}")
            else:
                load_string = loader + ' ' * (display_chunks - len(loader))
                print(f"[{load_string}]")
                loader += '.'

    if not verbose_output:
        print(f"Calcuated Accuracy: {final_acc:.2%}")

    return W1, b1, W2, b2

def make_predictions(X, W1 ,b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = get_predictions(A2)
    return predictions

def show_prediction(index, X, Y, W1, b1, W2, b2):
    '''
    This will display the image and prediction for user verification
    '''
    global verbose_output
    # None means it'll make a new column
    # to meet dimension needs of make_predictions
    vect_X = X[:, index, None]

    # Type: numpy.ndarray
    prediction = make_predictions(vect_X, W1, b1, W2, b2)

    # Type: numpy.unit8
    label = Y[index]

    if verbose_output:
        print(f"Prediction: {prediction}")
        print(f"Label: {label}")
        print('-'*20)

        current_image = vect_X.reshape((x, y)) * 255

        plt.gray()
        plt.imshow(current_image, interpolation = 'nearest')
        plt.show()

    return (int(prediction) == int(label))

def print_help():
    '''
    Prints help text for BabyNN.py
    '''
    output_text = ""
    output_text += "\n|" + "=" * 28 + " BabyNN.py Help " + "=" * 28 + "|\n"
    output_text += "|" + " " * 72 + "|\n"
    output_text += "| -h, --help       | Output help text" + " " * 36 + "|\n"
    output_text += "| -i, --iterations | Defines a custom number of iterations done" + " " * 10 + "|\n"
    output_text += "|                  | Default is 300. Limits itself to 10,000" + " " * 13 + "|\n"
    output_text += "| -f, --file       | Defines a custom filename to be used. Expects *.pkl" + " " + "|\n"
    output_text += "| -v, --verbose    | More information while the script is running" + " " * 8 + "|\n"
    output_text += "|" + "=" * 72 + "|\n"

    print(output_text)

def handle_input_args(input_args):
    '''
    Handle arguments passed in.
    See print_help() for information on accepted commands.
    Returns a dictionary for easy access.
    Prints errors for user information.
    '''
    global verbose_output
    #  Dictionary setup
    custom_changes = {}
    custom_changes['-h'] = False
    custom_changes['-f'] = "trained_params.pkl"
    custom_changes['-i'] = 300
    custom_changes['error'] = False
    #  Possible args that can be listed
    arg_set_help = {'-h', '--help'}
    arg_set_file = {'-f', '--file'}
    arg_set_iterations = {'-i', '--iterations'}
    arg_set_verbose = {'-v', '--verbose'}
    # Limits
    upper_iteration_limit = 1000

    #  User wants help output, no need for anything more to be done
    needs_help = any(element in arg_set_help for element in input_args)
    if needs_help:
        custom_changes['-h'] = True
        return custom_changes

    #  User wants a verbose output
    verbose_output = any(element in arg_set_verbose for element in input_args)

    #  If a custom file is defined
    new_file = any(element in arg_set_file for element in input_args)
    if new_file:
        try:
            new_file_pos = input_args.index(list(arg_set_file & set(input_args))[0]) + 1
            temp_value = input_args[new_file_pos]
            if temp_value[-4:] == '.pkl':
                custom_changes['-f'] = temp_value
            else:
                print(f"Invalid file type defined. Expected *.pkl | Received: *{temp_value[-4:]}")
                custom_changes['error'] = True
        except:
            print("No custom file name was provided with file arg set. Run with -h or --help for more information.")
            custom_changes['error'] = True

    #  If a new amount of iterations is defined
    new_iterations = any(element in arg_set_iterations for element in input_args)
    if new_iterations:
        try:
            new_iterations_pos = input_args.index(list(arg_set_iterations & set(input_args))[0]) + 1
            custom_changes['-i'] = int(input_args[new_iterations_pos])
            if custom_changes['-i'] > upper_iteration_limit:
                custom_changes['-i'] = upper_iteration_limit
                print(f"Automatically reducing number of iterations to {upper_iteration_limit}.")
        except:
            print(f"An integer value was expected for the new iterations value. Received: {input_args[new_iterations_pos]}")
            custom_changes['error'] = True

    return custom_changes

if __name__ == "__main__":
    #  Don't care about the first item in the list
    # input_args = list(ar for ar in sys.argv[1:] if len(sys.argv[1:]) > 0)
    custom_options = handle_input_args(list(ar for ar in sys.argv[1:] if len(sys.argv[1:]) > 0))
    if custom_options['-h']:
        print_help()
        quit()
    elif custom_options['error']:
        print("An error occurred with your parameters, aborting.")
        quit()

    #  Only import these afterwards when we know they're needed
    import pickle, time
    import numpy as np
    from matplotlib import pyplot as plt
    from keras.datasets import mnist

    file_name = custom_options['-f']
    iterations = custom_options['-i']

    # Read-in, format and normalise dataset
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data() #takes a while

    x = X_train.shape[1] # Pixel width of image should be 28
    y = X_train.shape[2] # Pixel height of image should be 28
    X_train = X_train.reshape(X_train.shape[0], x * y).T / 255 # Normalise
    X_test = X_test.reshape(X_test.shape[0], x * y).T  / 255 # Normalise

    #### MAIN ####
    # Time at start
    time_start = time.perf_counter()

    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.125, iterations)

    with open(file_name, "wb") as param_dump:
        pickle.dump((W1, b1, W2, b2), param_dump)

    with open(file_name, "rb") as param_dump:
        W1, b1, W2, b2 = pickle.load(param_dump)

    # Time at finish
    time_finish = time.perf_counter()

    image_set = {0, 1, 2, 100, 200}
    match_count = 0
    for i in image_set:
        if show_prediction(i, X_test, Y_test, W1, b1, W2, b2):
            match_count += 1

    print(f"\n{iterations} iterations took {time_finish-time_start:.5} seconds to complete.")
    print(f"Out of {len(image_set)} images, {match_count} were matches.")
    print(f"Overall Accuracy: {(match_count / len(image_set)):.0%}\n")
