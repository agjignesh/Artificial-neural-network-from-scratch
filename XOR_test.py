import ANN as n
import numpy as np
from keras.datasets import mnist


X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

model = n.NeuralNetwork(0.1, 1000, [3])
model.train(X, Y)


print(print(model.predict(X)))


print(model.evaluate(X, Y))