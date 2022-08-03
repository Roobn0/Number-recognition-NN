from tabnanny import verbose
import tensorflow as tf
import numpy as np
import os

from tensorflow import keras

# Load the model

HWD_model = tf.keras.models.load_model('Models\Model_HWD_1')

print("Available datasets are: HWD, Custom")
choose = input("Choose which dataset to use: ")

# Preparing test data

if choose == "HWD":

    HWD_data_test = np.load('Data\HWD\HWD_data_test.npy')
    HWD_labels_test = np.load('Data\HWD\HWD_label_test.npy')

    print("HWD data imported")

    # Evaluate the model or predict a digit
    print("Available options are: predict, evaluate")
    evaluate_predict = input("Evaluate or predict: ")

    if evaluate_predict == "evaluate":
        HWD_model.evaluate(HWD_data_test, HWD_labels_test, verbose=2)

    elif evaluate_predict == "predict":
        print(np.argmax(HWD_model.predict(HWD_data_test), axis=1))
        
    else:
        print("None of the available options were choosen")
        exit()

elif choose == "Custom":

    custom_data = np.load('Data\Custom\Processed data\data.npy')
    custom_labels = np.load('Data\Custom\Processed data\label.npy')

    print("Custom data imported")

    print("Available options are: predict, evaluate")
    evaluate_predict = input("Evaluate or predict: ")

    if evaluate_predict == "evaluate":
        HWD_model.evaluate(custom_data, custom_labels, verbose=2)

    elif evaluate_predict == "predict":
        print(np.argmax(HWD_model.predict(custom_data), axis=1))
        
    else:
        print("None of the available options were choosen")
        exit()

else:
    print("None of the available datasets were chosen")
    exit()
