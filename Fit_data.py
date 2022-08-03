import os
import numpy as np
import pandas as pd
import matplotlib
import tensorflow as tf

from matplotlib import pyplot as plt
from tensorflow import keras

# Laod the model

HWD_model = tf.keras.models.load_model('Models\Untrained_model_1')

#Preparing data

HWD_data_train = np.load('Data\HWD\HWD_data_train.npz')
HWD_labels_train = np.load('Data\HWD\HWD_label_train.npz')

print("Data prepared")

HWD_model.fit(HWD_data_train, HWD_labels_train, epochs=5)

print("Data fitted")