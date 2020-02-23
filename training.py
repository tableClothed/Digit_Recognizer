import tensorflow as tf
import pickle
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (a, b) = mnist.load_data()

# x_train = x_train/255.0
# x_train = tf.keras.utils.normalize(x_train, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])

model.fit(x_train, y_train, epochs=40)

model.save('MNIST.model')