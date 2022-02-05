'''
Handwritten digit classfication using TensorFlow and MNIST dataset. Generates a keras-compatible model scheme.

Author: Luzhou Zhang
'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Get MNIST dataset and load it into local variables
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize X-variables. Note we do not scale the y-axis because they are labels corresponding to the handwritten number,
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
# Add flat (1-dimensional) layer consisting of the 28x28 resolution images
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# Add 2 dense (all neurons connected to previous and next layer) convolutional hidden layers. They both use the relu linear rectification activator.
model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
# Add dense layer - converts output to a frequency that corresponds to the likelihood that each digit represents a specific number
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

# Compile model with Adam algorithm for stochastic gradient descent and a loss function dedicated to multiple categorical labels
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train model on data with 4 epochs. Testing has shown insignificant optimizations after 3 epochs.
model.fit(x_train, y_train, epochs=4)
accuracy, loss = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

model.save('digits.model')

''' DEPRECATED TESTING BLOCK. MAKE SURE TO RESHAPE YOUR DATA TO 28X28 BEFORE USING.
for x in range(1,6):
    img = cv.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'The result predicted is: {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
'''
