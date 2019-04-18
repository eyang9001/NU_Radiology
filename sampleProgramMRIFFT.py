# from Pillow import Image
import numpy
import pyfftw
import time

from MRI_FFT.OneD import Direct1d
from MRI_FFT.TwoD import Direct2d
from MRI_FFT.TwoD import OneDDecomp
from MRI_FFT.ThreeD import Direct3d
from MRI_FFT.ThreeD import TwoDDecomp

# create 2D MRI data
# -------------------------------------------
# Images:
# noface.jpg:           1024*1024
# noface_long.jpg:      1024*256
# noface_longmed.jpg:   1024*512
# noface_medium.jpg:    512*512
# noface_medsmall.jpg:  512*256
# noface_small.jpg:     256*256
#

print('Generating MRI Data')
# 1D data
data1d = numpy.random.rand(10).astype(numpy.complex64)
MRIdata1d = pyfftw.interfaces.numpy_fft.fft(data1d).astype('complex64')
# print(MRIdata1d.dtype)
# print(MRIdata1d.shape)
print('1D data generated')

# 2D data
# Convert to black and white
img = Image.open("noface_small.jpg").convert('L')
img.load()
# img.show()
a = numpy.array(img)
b = pyfftw.n_byte_align(a, 16)
MRIdata2d = pyfftw.interfaces.numpy_fft.fft2(b).astype('complex64')
# print(MRIdata2d.dtype)
# print(MRIdata2d.shape)
# # Generates a warning as complex data lost during conversion
# img = Image.fromarray(MRIdata.astype('uint8'))
# img.show()
print('2D data generated')

#3D data
img = Image.open("noface_small.jpg").convert('L')
img.load()
imgArray = numpy.array(img)
data3d = numpy.zeros((imgArray.shape[0], imgArray.shape[1], imgArray.shape[1]))
for i in range(0, MRIdata2d.shape[0]):
    data3d[i, :, :] = imgArray
b = pyfftw.n_byte_align(data3d, 16, dtype='complex64')
MRIdata3d = pyfftw.interfaces.numpy_fft.fftn(b)
# img = Image.fromarray(MRIdata3d[3, :, :].astype('uint8'))
# img.show()
# print(MRIdata3d.dtype)
# print(MRIdata3d.shape)
print('All Data Generated')
# -------------------------------------------


def oneD_direct():
    direct1dObject = Direct1d()
    result = direct1dObject.ifft1D(MRIdata1d)
    print('Original 1D data: ', data1d)
    print('After fft & ifft: ', result)


# Most efficient when all data has been collected in advance
def twoD_direct():
    direct2dObject = Direct2d(MRIdata2d.shape)
    start = time.time()

    # perform a timed 2d ifft
    result = direct2dObject.ifft2D(MRIdata2d)

    elapsed = time.time() - start
    print("Direct 2D Calculation Time: ", elapsed, "seconds", "\n")

    # img = Image.fromarray(result.astype('uint8'))
    # img.show()


# Most efficient when calculations can be performed during the scan
# Performs a 2D ifft using 1D decomposition across the second axis
def twoD_oneddecomp1():
    OneDDecompObject = OneDDecomp(MRIdata2d.shape, 1)
    start = time.time()
    # 1D iffts that can be completed during the scan: all but the last 1D array
    # send the data one line at a time
    for line in range(0, MRIdata2d.shape[1] - 1):
        # perform a timed 2d ifft
        OneDDecompObject.append1D(MRIdata2d[:, line])

    # scan completed: Record the scan time, then perform the last ifft pass
    # seperated from the for loop  so that it can be timed
    endOfScanTime = time.time()
    line = MRIdata2d.shape[1] - 1
    # Returns the 2D complex array
    result = OneDDecompObject.append1D(MRIdata2d[:, line])

    scanDuration = endOfScanTime - start
    finalPassDuration = time.time() - endOfScanTime
    calculationDuration = time.time() - start
    print("2D Transform with 1D Decomposition, axis 1")
    print("Minimum Scan Duration for maximum decomposition benefit: ", scanDuration, "seconds")
    print("Final ifft Pass Duration: ", finalPassDuration, "seconds")
    print("Total Calculation Time: ", calculationDuration, "seconds", "\n")

    # img = Image.fromarray(result.astype('uint8'))
    # img.show()


# Most efficient when calculations can be performed during the scan
# Performs a 2D ifft using 1D decomposition across the first axis
def twoD_oneddecomp0():
    OneDDecompObject = OneDDecomp(MRIdata2d.shape, 0)
    start = time.time()
    # 1D iffts that can be completed during the scan: all but the last 1D array
    # send the data one line at a time
    for line in range(0, MRIdata2d.shape[0] - 1):
        # perform a timed 2d ifft
        OneDDecompObject.append1D(MRIdata2d[line, :])

    # scan completed: Record the scan time, then perform the last ifft pass
    # seperated from the for loop  so that it can be timed
    endOfScanTime = time.time()
    line = MRIdata2d.shape[0] - 1
    # Returns the 2D complex array
    result = OneDDecompObject.append1D(MRIdata2d[line, :])

    scanDuration = endOfScanTime - start
    finalPassDuration = time.time() - endOfScanTime
    calculationDuration = time.time() - start
    print("2D Transform with 1D Decomposition, axis 0")
    print("Minimum Scan Duration for maximum decomposition benefit: ", scanDuration, "seconds")
    print("Final ifft Pass Duration: ", finalPassDuration, "seconds")
    print("Total Calculation Time: ", calculationDuration, "seconds", "\n")

    # img = Image.fromarray(result.astype('uint8'))
    # img.show()


def threeD_direct():
    direct3dObject = Direct3d(MRIdata3d.shape)
    start = time.time()

    # perform a timed 3d ifft
    result = direct3dObject.ifft3D(MRIdata3d)

    elapsed = time.time() - start
    print("Direct 3D Calculation Time: ", elapsed, "seconds", "\n")

    # img3d = Image.fromarray(result[3, :, :].astype('uint8'))
    # img3d.show()


def threeD_twoddecomp0():
    TwoDDecompObject = TwoDDecomp(MRIdata3d.shape, 0)
    start = time.time()

        # send the data one line at a time
    for line in range(0, MRIdata3d.shape[0] - 1):
        # perform a timed 2d ifft
        TwoDDecompObject.append2D(MRIdata3d[line, :, :])

    # scan completed: Record the scan time, then perform the last ifft pass
    # seperated from the for loop  so that it can be timed
    endOfScanTime = time.time()
    line = MRIdata3d.shape[0] - 1
    # Returns the 2D complex array
    result = TwoDDecompObject.append2D(MRIdata3d[line, :, :])

    scanDuration = endOfScanTime - start
    finalPassDuration = time.time() - endOfScanTime
    calculationDuration = time.time() - start
    print("3D Transform with 2D Decomposition, axis 0")
    print("Minimum Scan Duration for maximum decomposition benefit: ", scanDuration, "seconds")
    print("Final ifft Pass Duration: ", finalPassDuration, "seconds")
    print("Total Calculation Time: ", calculationDuration, "seconds", "\n")

    # img = Image.fromarray(result[line, :, :].astype('uint8'))
    # img.show()


def threeD_twoddecomp1():
    TwoDDecompObject = TwoDDecomp(MRIdata3d.shape, 1)
    start = time.time()

        # send the data one line at a time
    for line in range(0, MRIdata3d.shape[1] - 1):
        # perform a timed 2d ifft
        TwoDDecompObject.append2D(MRIdata3d[:, line, :])

    # scan completed: Record the scan time, then perform the last ifft pass
    # seperated from the for loop  so that it can be timed
    endOfScanTime = time.time()
    line = MRIdata3d.shape[1] - 1
    # Returns the 2D complex array
    result = TwoDDecompObject.append2D(MRIdata3d[:, line, :])

    scanDuration = endOfScanTime - start
    finalPassDuration = time.time() - endOfScanTime
    calculationDuration = time.time() - start
    print("3D Transform with 2D Decomposition, axis 1")
    print("Minimum Scan Duration for maximum decomposition benefit: ", scanDuration, "seconds")
    print("Final ifft Pass Duration: ", finalPassDuration, "seconds")
    print("Total Calculation Time: ", calculationDuration, "seconds", "\n")

    # img = Image.fromarray(result[line, :, :].astype('uint8'))
    # img.show()


def threeD_twoddecomp2():
    TwoDDecompObject = TwoDDecomp(MRIdata3d.shape, 2)
    start = time.time()

        # send the data one line at a time
    for line in range(0, MRIdata3d.shape[2] - 1):
        # perform a timed 2d ifft
        TwoDDecompObject.append2D(MRIdata3d[:, :, line])

    # scan completed: Record the scan time, then perform the last ifft pass
    # seperated from the for loop  so that it can be timed
    endOfScanTime = time.time()
    line = MRIdata3d.shape[2] - 1
    # Returns the 2D complex array
    result = TwoDDecompObject.append2D(MRIdata3d[:, :, line])

    scanDuration = endOfScanTime - start
    finalPassDuration = time.time() - endOfScanTime
    calculationDuration = time.time() - start
    print("3D Transform with 2D Decomposition, axis 2")
    print("Minimum Scan Duration for maximum decomposition benefit: ", scanDuration, "seconds")
    print("Final ifft Pass Duration: ", finalPassDuration, "seconds")
    print("Total Calculation Time: ", calculationDuration, "seconds", "\n")

    # img = Image.fromarray(result[line, :, :].astype('uint8'))
    # img.show()

if __name__ == "__main__":
    oneD_direct()

    twoD_direct()
    twoD_oneddecomp0()
    twoD_oneddecomp1()

    threeD_direct()
    threeD_twoddecomp0()
    threeD_twoddecomp1()
    threeD_twoddecomp2()