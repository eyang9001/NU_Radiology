from sys import platform as sys_pf
# nilearn library for 3d visualizations: https://nilearn.github.io/plotting/index.html#adding-overlays-edges-contours-contour-fillings-markers-scale-bar
# if sys_pf == 'darwin':
#     import matplotlib
#     matplotlib.use("TkAgg")
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
from sklearn.model_selection import train_test_split

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

image_shape = (256, 256, 176)


# Read in transformed images
def read_file(filename):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(filename)
    image = reader.Execute()

    # img1 = sitk.ReadImage(folder + item)  # alternative way to pull in image

    # convert image into np array & perform fft
    img = sitk.GetArrayFromImage(image)
    # Transpose the image so the first axis is Anterior-Posterior
    return img


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


def visualize3(orig, blur, predict):
    plt.figure(figsize=(30, 30))
    plt.subplot(131), plt.imshow(orig, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(blur, cmap='gray')
    plt.title('Blurred'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(predict, cmap='gray')
    plt.title('Model Output'), plt.xticks([]), plt.yticks([])


def visualize_model(model, ind, X, y, save=False, fn=''):
    x_input = X[ind]
    out = model.predict(np.array(x_input[np.newaxis, ...]))
    visualize3(y[ind][..., 0], x_input[..., 0], out[0, ..., 0])
    if save:
        plt.savefig(fn)


def generate_data_3d(t_filenames, o_filenames):
    # Takes in the files of the transformed images and the originals to create the pairs
    # Also sets the min and max slice numbers to take in
    x_data = []
    y_data = []
    for i in range(len(t_filenames)):
        print('Generating data for: ' + t_filenames[i])
        x_mri = read_file(t_filenames[i])
        y_mri = read_original(o_filenames[i])
        x_data.append(
            (x_mri - np.amin(x_mri)) / (np.amax(x_mri) - np.amin(x_mri)))  # normalizes the intensity to between 0 and 1
        y_data.append((y_mri - np.amin(y_mri)) / (np.amax(y_mri) - np.amin(y_mri)))
    # Shape of x & y will be [# mris, 256, 256, 176, 1]
    x_data = np.array(x_data)[..., np.newaxis]
    y_data = np.array(y_data)[..., np.newaxis]
    return x_data, y_data


# Simple 3d unet with mse loss

def conv_layer(filts, dim):
    # abstracted a single conv layer out since the parameters outside of dimension were kept the same
    return Conv3D(filts, dim, activation='relu', padding='same', kernel_initializer='he_normal')


# u_net model
def unet_model(lr=1e-4, input_size=(256, 256, 176, 1), dropout_level=0.1):
    inputs = Input(input_size)

    conv1 = conv_layer(64, 3)(inputs)
    conv1 = conv_layer(64, 3)(conv1)
    drop1 = Dropout(dropout_level)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(drop1)
    conv2 = conv_layer(128, 3)(pool1)
    conv2 = conv_layer(128, 3)(conv2)
    drop2 = Dropout(dropout_level)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(drop2)
    conv3 = conv_layer(256, 3)(pool2)
    conv3 = conv_layer(256, 3)(conv3)
    drop3 = Dropout(dropout_level)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(drop3)
    conv4 = conv_layer(512, 3)(pool3)
    conv4 = conv_layer(512, 3)(conv4)
    drop4 = Dropout(dropout_level)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)
    conv5 = conv_layer(1024, 3)(pool4)
    conv5 = conv_layer(1024, 3)(conv5)
    drop5 = Dropout(dropout_level)(conv5)

    # Decoder
    up6 = conv_layer(512, 2)(UpSampling3D(size=(2, 2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=4)
    drop6 = Dropout(dropout_level)(merge6)
    conv6 = conv_layer(512, 3)(drop6)
    conv6 = conv_layer(512, 3)(conv6)

    up7 = conv_layer(256, 2)(UpSampling3D(size=(2, 2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=4)
    drop7 = Dropout(dropout_level)(merge7)
    conv7 = conv_layer(256, 3)(drop7)
    conv7 = conv_layer(256, 3)(conv7)

    up8 = conv_layer(128, 2)(UpSampling3D(size=(2, 2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=4)
    drop8 = Dropout(dropout_level)(merge8)
    conv8 = conv_layer(128, 3)(drop8)
    conv8 = conv_layer(128, 3)(conv8)

    up9 = conv_layer(64, 2)(UpSampling3D(size=(2, 2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=4)
    drop9 = Dropout(dropout_level)(merge9)
    conv9 = conv_layer(64, 3)(drop9)
    conv9 = conv_layer(64, 3)(conv9)
    conv9 = conv_layer(2, 3)(conv9)
    conv10 = Conv3D(1, 1, activation='linear')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model
# generate 3d training data
trans_folder = '../transforms/'
orig_folder = '../originals/'
X_filenames = []
y_filenames = []
for item in os.listdir(trans_folder):
    if item.endswith(".nii"):
        X_filenames.append(trans_folder + item)
        y_filenames.append(orig_folder + item[:-12] + '.nii')

# X, y = generate_data(['gdrive/My Drive/NU_Rad/transforms/M02_motion5_trans.txt.gz'], ['gdrive/My Drive/NU_Rad/mris/M02.nii'])

# limiting # of files to train on:
lim = 10
X_train, y_train = generate_data_3d(X_filenames[:lim], y_filenames[:lim])

print('Data Generated')
print(np.shape(X_train))

# Checkpoint
check_path = 'gdrive/My Drive/NU_Rad/models/3d_unet_mse_model'
checkpoint = ModelCheckpoint(check_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model = unet_model()
model.fit(X_train, y_train, batch_size=1, validation_split=0.1, epochs=3, use_multiprocessing=True, callbacks=callbacks_list, verbose=1)