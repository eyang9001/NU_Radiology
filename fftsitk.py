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

for item in os.listdir(folder):
    if item.endswith(".gz"):
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(folder + item)
        image = reader.Execute()

        # img1 = sitk.ReadImage(folder + item)  # alternative way to pull in image

        # show image
        nda = sitk.GetArrayFromImage(image)
        plt.imshow(nda)
        plt.show()
