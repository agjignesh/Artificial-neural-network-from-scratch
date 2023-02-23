import ANN 
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


(train_X, train_y), (test_X, test_y) = mnist.load_data()


train_X = train_X/255
test_X = test_X/255

train_X = train_X.reshape(train_X.shape[0], 784, 1)
test_X = test_X.reshape(test_X.shape[0], 784, 1)

train_y = train_y.reshape(train_y.shape[0],1)
test_y = test_y.reshape(test_y.shape[0],1)

train_y_new = np.zeros(shape = (train_y.shape[0],10,1))
for i in range(train_y.shape[0]):
    train_y_new[i][int(train_y[i][0])][0] = 1
train_y = train_y_new

test_y_new = np.zeros((test_y.shape[0],10,1))
for i in range(test_y.shape[0]):
    test_y_new[i][int(test_y[i][0])][0] = 1
test_y = test_y_new

model = ANN.NeuralNetwork(0.001, 20)
model.add(784, 128, 'relu')
model.add(128,32, 'relu')
model.add(32, 10, 'softmax')

model.train(train_X[::10], train_y[::10], 10)

print(100*model.predict([test_X[1]]))
print(test_y[1])
