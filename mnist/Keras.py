# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Keras tutorial
# Tutorial from [here](https://keras.io/examples/vision/mnist_convnet/).

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# ## Load data

# +
num_classes = 10
input_shape = (28,28,1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to [0,1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# -

x_train.shape, y_train.shape

# +
# Change input shape
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

x_train.shape, x_test.shape
# -

y_test

# Encode classes as vectors
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

y_train

# +
## Model

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3,3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, kernel_size=(3,3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes,activation="softmax")
        
    ]
)
# -

model.summary()

# +
batch_size = 128

epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# +
score = model.evaluate(x_test, y_test, verbose=0)

print("Test loss:", score[0])
print("Test accuracy:", score[1])
# -


