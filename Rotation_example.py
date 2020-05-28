from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
# ! pip install simpleitk

import os
import numpy as np
# import nibabel as nib
from matplotlib import pyplot as plt
import matplotlib
import SimpleITK as sitk
from scipy import ndimage
import random

folder = 'mris/'

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

        # FFT in 3d
        v = np.fft.fftn(img)

back_fft2 = v.copy()
back_real = back_fft2.real
back_imag = back_fft2.imag

for i in range(100):
    trans = random.randint(-50,50)
    rot = random.randint(-50,50)
    back_slice_r = back_real[i]
    back_slice_i = back_imag[i]

    # Testing translation
    back_rot_r = ndimage.shift(back_slice_r,[trans,0], mode='constant', cval=100)
    back_rot_i = ndimage.shift(back_slice_i,[trans,0], mode='constant', cval=100)

    # Testing rotation
    back_rot_r = ndimage.rotate(back_slice_r, rot, reshape = False)
    back_rot_i = ndimage.rotate(back_slice_i, rot, reshape = False)

    back_rot = back_rot_r + back_rot_i * 1j
    back_fft2[i] = back_rot


back_vis = np.fft.ifftn(back_fft2)
back_display = back_vis.astype(int)

plt.subplot(121), plt.imshow(orig_slice, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(back_display[100], cmap='gray')
plt.title('Blurred'), plt.xticks([]), plt.yticks([])
plt.show()