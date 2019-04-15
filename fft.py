# from Pillow import Image
import os
import numpy as np
import nibabel as nib
from MRI_FFT.TwoD import Direct2d

folder = 'data/'

# Goes through all of the files that end in .gz
for item in os.listdir(folder):
    if item.endswith(".gz"):
        img = nib.load(folder + item)
        # convert the image into a numpy array
        a = np.array(img.dataobj)
        print(a)

        # do the reverse FFT
        direct2dobject = Direct2d(a.shape)
        result = direct2dobject.ifft2D(a)

        # # create new image
        # newimg = Image.fromarray(result.astype('uint8'))
        # newimg.show()