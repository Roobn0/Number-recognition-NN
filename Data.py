import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

from tensorflow import keras 

#500x500 data (HWD)

print("START")

HWD_data = np.load('Data\Images500.npy')
HWD_labels = np.load('Data\Labels.npy')

print("LABEL")

HWD_label = [label[0] for label in HWD_labels]
HWD_label = np.array(HWD_label)

print("DATA")

(HWD_data_train, HWD_label_train) = (HWD_data[:10000], HWD_label[:10000])
(HWD_data_test, HWD_label_test) = (HWD_data[:10000], HWD_label[:10000])

print("RESIZE")

HWD_data_train = np.invert(HWD_data_train)
HWD_data_test = np.invert(HWD_data_test)

HWD_data_train, HWD_data_test = HWD_data_train / 255.0, HWD_data_test / 255.0

print("SAVE-HWD-DATA")

np.savez_compressed('Data\HWD\HWD_data_train', HWD_data_train)
np.savez_compressed('Data\HWD\HWD_label_train', HWD_label_train)

print("SAVE-HWD-LABEL")

np.savez_compressed('Data\HWD\HWD_data_test', HWD_data_test)
np.savez_compressed('Data\HWD\HWD_label_test', HWD_label_test)

print("SAVE-MNIST")

#25x25 data (MNIST)

mnist = keras.datasets.mnist

(mnist_data_train, mnist_label_train), (mnist_data_test, mnist_label_test) = mnist.load_data()
mnist_data_train, mnist_data_test = mnist_data_train / 255.0, mnist_data_test / 255.0

np.save('Data\MNIST\MNIST_data_train', mnist_data_train)
np.save('Data\MNIST\MNIST_label_train', mnist_label_train)

np.savez('Data\MNIST\MNIST_data_test', mnist_data_test)
np.save('Data\MNIST\MNIST_label_test', mnist_label_test)

print("FINISHED")