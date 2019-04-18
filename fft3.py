from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

# import cv2
import os
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt


folder = 'data/'

# Goes through all of the files that end in .gz
for item in os.listdir(folder):
    if item.endswith(".gz"):
        img = nib.load(folder + item)
        # convert the image into a numpy array
        a = np.array(img.dataobj)
        print(a)
        print(a.shape)

        f = np.fft.fft2(a)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))

        # plt.subplot(121), plt.imshow(img, cmap='gray')
        # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
        # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        # plt.show()
