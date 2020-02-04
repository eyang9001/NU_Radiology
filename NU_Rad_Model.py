from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import SimpleITK as sitk
from scipy import ndimage
import random
import math

import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Concatenate, Add, Average, Input, Dense, Flatten, BatchNormalization, Activation, LeakyReLU, Reshape
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D
from tensorflow.keras import backend as K
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.optimizers as optimizers
from sklearn import preprocessing as prepro

#   Preprocessing
# Data- Slice range from 70 to 220
# Flatten each image
# keras.utils.normalize each image
# reshape back

#   Architecture

# Input layer

# 2d convolution
# activation- relu
# 2d convlution
# activation- relu

# loss = mse


# TO DO-
# Double check the normalization function by multiplying output slice by max val & visualizing
# Put together full pipeline taking in all mris


image_shape = (256, 256, 176)

# Read in transformed images
def read_file(filename):
    data = np.loadtxt(filename, delimiter=",")
    data = np.reshape(data, image_shape)
    return data

# Read in non-transformed images
def read_original(filename):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(filename)
    image = reader.Execute()

    # img1 = sitk.ReadImage(folder + item)  # alternative way to pull in image

    # convert image into np array & perform fft
    img = sitk.GetArrayFromImage(image)
    # Transpose the image so the first axis is Anterior-Posterior
    img = np.transpose(img, (2, 1, 0))
    return img

def visualize(orig, new):
    plt.figure(figsize=(20, 20))
    plt.subplot(121), plt.imshow(orig, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(new, cmap='gray')
    plt.title('New'), plt.xticks([]), plt.yticks([])
    plt.show()


def visualize3(orig, blur, predict):
    plt.figure(figsize=(30, 30))
    plt.subplot(131), plt.imshow(orig, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(blur, cmap='gray')
    plt.title('Blurred'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(predict, cmap='gray')
    plt.title('Predicted'), plt.xticks([]), plt.yticks([])

def generate_data(t_filenames, o_filenames, min_slc = 70, max_slc = 220):
    # Takes in the files of the transformed images and the originals to create the pairs
    # Also sets the min and max slice numbers to take in
    x_data = []
    y_data = []
    for i in range(len(t_filenames)):
        print('Generating data for: ' + t_filenames[i])
        x_mri = read_file(t_filenames[i])
        y_mri = read_original(o_filenames[i])
        for ii in np.arange(min_slc, max_slc+1):
            x_data.append(x_mri[ii])
            y_data.append(y_mri[ii])
    # Shape of x & y will be [# slices, 256, 176]
#     x_data = norm_slices(x_data)  # for normalizing
#     y_data = norm_slices(y_data)  # for normalizing
    return x_data, y_data

# Normalizes each image slice
def norm_slices (data):
    orig_shape = np.shape(data)
    new_shape = np.reshape(data, (list(orig_shape)[0], list(orig_shape)[1] * list(orig_shape)[2]))
    new_shape_norm = tf.keras.utils.normalize(new_shape, axis=0, order=2)
    return np.reshape(new_shape_norm, orig_shape)

# pulling in images

trans_folder = 'gdrive/My Drive/NU_Rad/transforms/'
orig_folder = 'gdrive/My Drive/NU_Rad/mris/'
X_filenames = []
y_filenames = []
for item in os.listdir(trans_folder):
    if item.endswith("_trans.txt.gz"):
        X_filenames.append(trans_folder + item)
        y_filenames.append(orig_folder + item[:3] + '.nii')

# X, y = generate_data(['gdrive/My Drive/NU_Rad/transforms/M02_motion5_trans.txt.gz'], ['gdrive/My Drive/NU_Rad/mris/M02.nii'])
X, y = generate_data(X_filenames, y_filenames)

# Model

X = np.reshape(X, (3473, 256, 176, 1))
y = np.reshape(y, (3473, 256, 176))
model = Sequential()
model.add(Conv2D(64, (5,5), padding='same', input_shape = X.shape[1:]))
model.add(Activation("relu"))

model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation("relu"))

model.add(Dense(64))
model.add(Dense(1))
model.add(Reshape((256,176)))

model.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])
model.fit(X, y, batch_size=10, validation_split=0.1, epochs = 10)

# Predict

X_predict, y_predict = generate_data(['gdrive/My Drive/NU_Rad/transforms/M02_motion5_trans.txt.gz'], ['gdrive/My Drive/NU_Rad/mris/M02.nii'])

# Visualize

predict_slice = 120
X_input = np.reshape(X_predict[predict_slice], (1, 256, 176, 1))
predict_out = model.predict(np.array(X_input))

output = np.reshape(predict_out, (256, 176))
# output = output*690
visualize3(y_predict[predict_slice], X_predict[predict_slice], output)