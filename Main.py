from typing import no_type_check
from unicodedata import digit
import tensorflow as tf
import numpy as np
import os

from tensorflow import keras

#Sequential number recognition model

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(500, 500)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

print("Model created")

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Model compiled")

# Save the model

model.save('Models\Untrained_model_1')

print("Model saved")
