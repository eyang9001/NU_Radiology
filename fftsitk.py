from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

import os
import numpy as np
# import nibabel as nib
from matplotlib import pyplot as plt
import matplotlib
import SimpleITK as sitk

folder = 'data/'

def fft_2d():
    for item in os.listdir(folder):
        if item.endswith(".gz"):
            reader = sitk.ImageFileReader()
            reader.SetImageIO("NiftiImageIO")
            reader.SetFileName(folder + item)
            image = reader.Execute()

            # img1 = sitk.ReadImage(folder + item)  # alternative way to pull in image

            # convert image into np array & perform fft
            img = sitk.GetArrayFromImage(image)
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift))

            # display the slice & K-Space
            plt.subplot(121), plt.imshow(img, cmap='gray')
            plt.title('MRI Slice'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
            plt.title('K-Space'), plt.xticks([]), plt.yticks([])
            plt.show()

def fft_3d():
    for item in os.listdir(folder):
        if item.endswith(".nii"):
            reader = sitk.ImageFileReader()
            reader.SetImageIO("NiftiImageIO")
            reader.SetFileName(folder + item)
            image = reader.Execute()

            # img1 = sitk.ReadImage(folder + item)  # alternative way to pull in image

            # convert image into np array & perform fft
            img = sitk.GetArrayFromImage(image)
            # print(img.shape)
            orig_slice = img[100]
            v = np.fft.fftn(img)
            new_slice = v[100]
            # print(v.shape)
            # print(v[0][0])
            magnitude_spectrum = 20 * np.log(np.abs(new_slice))
            back_spectrum = np.fft.fftn(magnitude_spectrum)
            back_img = 20 * np.log(np.abs(back_spectrum))

            back_max = np.amax(back_spectrum)
            back_min = np.amin(back_spectrum)

            back_fft = np.fft.fftn(v)
            back_slice = back_fft[100]

            print(np.amax(orig_slice))

            # plt.subplot(121), plt.imshow(back_img, cmap='gray')
            # plt.title('MRI Slice'), plt.xticks([]), plt.yticks([])
            plt.subplot(121), plt.imshow(orig_slice, cmap='gray')
            plt.title('MRI Slice'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
            plt.title('K-Space'), plt.xticks([]), plt.yticks([])


            plt.show()



fft_3d()