import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(trainX, trainY), (a, b) = mnist.load_data()

trainX = trainX[:1000]
trainY = trainY[:1000]

data = trainX[0]
label = trainY[0]
pixels = data.reshape((28, 28))

plt.imread(pixels)
plt.show()