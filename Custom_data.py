import os
import numpy as np
import pandas as pd
import matplotlib
import PIL
import tensorflow as tf

from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from tkinter import Image
from tensorflow import keras
 
print("START")

def image_resize(width, height, image_path, labels, save_data_path, save_label_path):
    i = 0
    custom_labels = np.array(labels)

    for file in os.listdir(image_path):
        
        file_path = os.path.join(image_path, file)
        custom_image = PIL.Image.open(file_path)
        custom_image = PIL.ImageOps.grayscale(custom_image)
        custom_image = custom_image.resize((width, height), PIL.Image.ANTIALIAS)
        data = np.invert([np.asarray(custom_image)]) / 255.0

        if i == 0:
            custom_data = data

        else:
            custom_data = np.append(custom_data, data, axis=0)

        i += 1

    np.save(save_data_path, custom_data)
    np.save(save_label_path, custom_labels)


image_path = 'Data\Custom\Custom images'
save_data_path = 'Data\Custom\Processed data\data.npy'
save_label_path = 'Data\Custom\Processed data\label.npy'
label = [0,1,2,3,4]

# Create custom data

image_resize(500, 500, image_path, label, save_data_path, save_label_path)

print("Data saved")