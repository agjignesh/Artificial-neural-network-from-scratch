import numpy as np
import matplotlib.pyplot as plt

class dense:

    def __init__(self,no_of_inputs, no_of_output):
        # intial weights and bias as random values
        self.weights = np.random.rand(no_of_output, no_of_inputs) - 0.5
        self.bias = np.random.rand(no_of_output, 1) - 0.5
    
    def forward(self, inputs):
        # input is a column vector
        # gives the output of the layer
        self.inputs = inputs
        self.output = np.dot(self.weights, inputs) + self.bias
    
    def backward(self, derivative_error_to_output, learning_rate):
        # calculates the derivative of the error with respect to the weights and bias and learns
        derivative_error_to_weights = np.dot(derivative_error_to_output, self.inputs.T)
        derivative_error_to_bias = derivative_error_to_output
        self.weights -= learning_rate * derivative_error_to_weights
        self.bias -= learning_rate * derivative_error_to_bias
        derivative_error_to_inputs = np.dot(self.weights.T, derivative_error_to_output)
        return derivative_error_to_inputs

class activation:

    def __init__(self, activation, activation_derivative):
        # takes the activation function and its derivative can be tanh, sigmoid, relu, leaky relu, etc
        self.activation = activation
        self.activation_derivative = activation_derivative
    
    def forward(self, inputs):
        # gives the output of the layer
        self.inputs = inputs
        self.output = self.activation(inputs)

    def backward(self, derivative_error_to_output, learning_rate):
        # calculates the derivative of the error with respect to the inputs and learns
        return self.activation_derivative(self.inputs) * derivative_error_to_output
    
# These are the three types of activation function we have 
class Tanh(activation):
    # when we create an object of this class, it will automatically call the __init__ function of the parent class
    # and we can use the functions of the parent class
    def __init__(self):

        def tanh(x):
            return np.tanh(x)
        
        def tanh_derivative(x):
            return 1 - np.tanh(x)**2
        
        super().__init__(tanh, tanh_derivative)

class Sigmoid(activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

class Softmax():
    def forward(self, input):
        tmp = np.exp(input - np.max(input))
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        # This version is faster than the one presented in the video
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
        # Original formula:
        # tmp = np.tile(self.output, n)
        # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)

class Relu(activation):
    
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)
        
        def relu_derivative(x):
            return x > 0
        
        super().__init__(relu, relu_derivative)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)



class NeuralNetwork:

    def __init__(self, Learning_rate, Epoch):
        """Enter the learning rate and the number of epochs"""
        self.epoch = Epoch
        self.learning_rate = Learning_rate
        self.NN =[]


    def add(self, no_of_input, no_of_output, activation_type):
        """Use this function to add layers to the neural network and specify the activation function"""
        if activation_type == 'tanh':
            self.NN.append(dense(no_of_input, no_of_output))
            self.NN.append(Tanh())
        elif activation_type == 'sigmoid':
            self.NN.append(dense(no_of_input, no_of_output))
            self.NN.append(Sigmoid())
        elif activation_type == 'softmax':
            self.NN.append(dense(no_of_input, no_of_output))
            self.NN.append(Softmax())
        elif activation_type == 'relu':
            self.NN.append(dense(no_of_input, no_of_output))
            self.NN.append(Relu())
        else:
            print('Activation type not supported')
        
    def train(self, X_train, Y_train,no_of_output):
        """Enter the training data and the labels in the form of a matrix where each column is a data point"""
        # training the neural network
        error = []
        for i in range(self.epoch):

            for x,y in zip(X_train, Y_train):

                # forward propagation
                self.NN[0].forward(x)
                for j in range(1, len(self.NN)):
                    self.NN[j].forward(self.NN[j-1].output)
                

                # backward propagation
                error.append(mse(y, self.NN[-1].output))
                error_derivative = mse_derivative(y, self.NN[-1].output)
                for j in range(len(self.NN) - 1, -1, -1):
                    error_derivative = self.NN[j].backward(error_derivative, self.learning_rate)
            print(100 * (i + 1) / self.epoch, '%' , 'completed')

    def predict(self, X_test):
        """Enter the test data in the form of a matrix where each column is a data point"""
        # each data point is a column in the matrix X_test
        # returns the predicted value for each data point
        Y_pred = []
        for x in X_test:
            self.NN[0].forward(x)
            for j in range(1, len(self.NN)):
                self.NN[j].forward(self.NN[j-1].output)
            Y_pred.append(self.NN[-1].output)
        return np.array(Y_pred)
    
    def evaluate(self, X_test, Y_test):
        # each data point is a column in the matrix X_test
        # each label is a column in the matrix Y_test
        # returns the mean squared error

        Y_pred = self.predict(X_test)
        return  mse(Y_test, Y_pred)
