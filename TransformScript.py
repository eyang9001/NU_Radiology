from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

# ! pip install simpleitk
import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import math
from argprase import ArgumentParser

# mris_folder = 'NU_Radiology/mris/'
# motion_folder = 'NU_Radiology/motions/'
image_shape = (256,256,176)

# Flags- mri folder, motion folder, slice rate,

argparser = ArgumentParser()
args_group = argparser.add_argument_group(title='Running args')
args_group.add_argument('-mrifolder', type=str, help='Set this argument to the folder path of the mris to be transformed, default is /mris', required=False, default="/mris/")
args_group.add_argument('-motionfolder', type=str, help='Set this argument to the folder path of the mris to be transformed, default is /motion', required=False, default="/motion/")
args_group.add_argument('-outfolder', type=str, help='Set this argument to the folder path to store the output, default is /transforms', required=False, default="/transforms/")
args_group.add_argument('-slicerate', type=int, help='Slice rate of transformations (1 is all slices are transformed, 2 is every other is transformed), default is 1', required=False, default=1)
args = argparser.parse_args()

mris_folder = args.mrifolder
motion_folder = args.motionfolder
out_folder = args.outfolder
slice_rate = args.slicerate

# mris_folder = 'gdrive/My Drive/NU_Rad/mris/'            # UPDATE THE FLAG FOR THIS
# motion_folder = 'gdrive/My Drive/NU_Rad/motion/'        # UPDATE THE FLAG FOR THIS
# cache_folder = 'gdrive/My Drive/NU_Rad/cache/'
# out_folder = 'gdrive/My Drive/NU_Rad/transforms/'

class mri(object):
    def __init__(self, filename):
        self.conv_file(filename)
        self.ks = np.fft.fftn(self.original)
        self.ks = np.fft.fftshift(self.ks)
        self.max = np.amax(self.original)
        self.min = np.amin(self.original)
        self.ks_vis = np.abs(self.ks)
        self.x, self.y, self.z = self.original.shape
        self.slices = self.x
        self.rate = slice_rate  # The number of slices that are skipped for transformations

    def conv_file(self, filename):
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(filename)
        image = reader.Execute()

        # img1 = sitk.ReadImage(folder + item)  # alternative way to pull in image

        # convert image into np array & perform fft
        img = sitk.GetArrayFromImage(image)
        # Transpose the image so the first axis is Anterior-Posterior
        img = np.transpose(img, (2, 1, 0))
        self.original = img

    def fft_back(self, kspace):
        # Produce the image given the kspace
        back_fft = np.fft.ifftn(kspace)
        return back_fft

    # modifies the whole 3d kspace and then returns a specific slice
    # returns the slice of the modified kspace as well as visual representation
    def mod_kspace_slice(self, trans, rot, slice_num):
        # Rotations are passed in as [yaw, pitch, roll]
        # rotation axes of (1,2) is nodding,(0,2) is shaking, (0,1) is cracking
        mod3d = self.ks.copy()
        mod_v = self.original.copy()
        mod_r = mod3d.real
        mod_i = mod3d.imag

        if rot != None:
            # Yaw rotation
            mod_v = ndimage.rotate(mod_v, rot[0], (0, 2), reshape=False)
            mod_r = ndimage.rotate(mod_r, rot[0], (0, 2), reshape=False)
            mod_i = ndimage.rotate(mod_i, rot[0], (0, 2), reshape=False)
            # Pitch rotation
            mod_v = ndimage.rotate(mod_v, rot[1], (1, 2), reshape=False)
            mod_r = ndimage.rotate(mod_r, rot[1], (1, 2), reshape=False)
            mod_i = ndimage.rotate(mod_i, rot[1], (1, 2), reshape=False)
            # Roll rotation
            mod_v = ndimage.rotate(mod_v, rot[2], (0, 1), reshape=False)
            mod_r = ndimage.rotate(mod_r, rot[2], (0, 1), reshape=False)
            mod_i = ndimage.rotate(mod_i, rot[2], (0, 1), reshape=False)
        if trans != None:
            mod_v = ndimage.shift(mod_v, trans, mode='constant', cval=0)
            mod_r = ndimage.shift(mod_r, trans, mode='constant', cval=0)
            mod_i = ndimage.shift(mod_i, trans, mode='constant', cval=0)
        return mod_r[slice_num] + mod_i[slice_num] * 1j, mod_v[slice_num]

    # Translating a specific slice
    # Returns the slice after translation
    def translate(self, space, slice_num, value):
        if space == 'v':
            slc = self.original[slice_num]
        elif space == 'k':
            slc = self.ks_vis[slice_num]
        tran = ndimage.shift(slc, value, mode='constant', cval=0)
        return tran

    # Rotating a specific slice
    # Returns the slice after rotation
    def rotate(self, space, slice_num, value, axis=None):
        # If the rotation is in visual space, or k-space
        if space == 'v':
            slc = self.original[slice_num]
            rot = ndimage.rotate(slc, value, reshape=False)
        elif space == 'k':
            slc = self.ks[slice_num]
            rot = ndimage.rotate(slc, value, reshape=False, cval=0)
        return rot

    def create_mod_slices(self, motion):
        modded_k = []
        modded_v = []
        trans_log = []

        for i in np.arange(len(self.ks)):
            motion_step = int(i * motion.length / len(self.ks))
            if (i % self.rate == 0) and motion_step < motion.length:
                trans = [motion.trans_x[motion_step], motion.trans_y[motion_step], motion.trans_z[motion_step]]
                rot = [motion.yaw[motion_step], motion.pitch[motion_step], motion.roll[motion_step]]
                k, v = self.mod_kspace_slice(trans, rot, i)
                modded_k.append(k)
                modded_v.append(v)
                trans_log.append(
                    [i, motion.trans_x[motion_step], motion.trans_y[motion_step], motion.trans_z[motion_step],
                     motion.yaw[motion_step], motion.pitch[motion_step], motion.roll[motion_step]])
                print('Slice #{} out of {} completed'.format(i, len(self.ks)))
            else:
                modded_k.append(self.ks[i])
                modded_v.append(self.original[i])
            # Here, the modded_v is only a visual representation of each slice's transformation
            # Returns transf_v as the blurred images
        transf_v = mri1.fft_back(modded_k)
        transf_v = self.transp_img(np.abs(transf_v))

        # transf_v will be left-right orientation while modded_v is anterior-posterior
        return transf_v, modded_v, trans_log

    def transp_img(self, data):
        return np.transpose(data, (2, 1, 0))

    def save_to_file(self, data, name):
        # Takes in 3d array and converts to 2d, then saves to file
        # If the file doesn't have imaginary component, set 'frm' as 'v'
        output_r = []
        for sublist in data:
            for item in sublist:
                output_r.append(item)
        np.savetxt(name, output_r, fmt="%s", delimiter=',')


class motion(object):
    def __init__(self, filename):
        trans_scale = 1
        self.filename = filename
        self.raw = np.loadtxt(filename)
        self.length = len(self.raw)
        self.yaw = []
        self.pitch = []
        self.roll = []
        self.trans_x = []
        self.trans_y = []
        self.trans_z = []
        for i in range(len(self.raw)):
            self.yaw.append(self.raw[i][0] * 180 / math.pi)
            self.pitch.append(self.raw[i][1] * 180 / math.pi)
            self.roll.append(self.raw[i][2] * 180 / math.pi)
            # Still need to figure out the conversion of translations from motion file
            self.trans_x.append(self.raw[i][3] * trans_scale)
            self.trans_y.append(self.raw[i][4] * trans_scale)
            self.trans_z.append(self.raw[i][5] * trans_scale)


def read_file(filename):
    data = np.loadtxt(filename, delimiter=",")
    data = np.reshape(data, (176, 256, 256))
    return data


# Full pipeline


for item in os.listdir(mris_folder):
    if item.endswith(".nii"):
        mri1 = mri(mris_folder + item)
        for mot_file in os.listdir(motion_folder):
            suffix = '/' + item[:-4] + '_' + mot_file[:-4]
            motion1 = motion(motion_folder + mot_file)

            print('Processing mri: {}, motion: {}'.format(item, mot_file))
            transf_v, _, mod_log = mri1.create_mod_slices(motion1)

            np.savetxt(out_folder + suffix + '_1mod.txt', mod_log, fmt="%s", delimiter=',')
            print('Cache saved for mri: {}, motion: {}'.format(item, mot_file))

            mri1.save_to_file(transf_v, out_folder + suffix + '_trans_1mod.txt.gz')
            print('Modified vis saved for mri: {}, motion: {}'.format(item, mot_file))
#             os.remove(cache_folder + suffix + '_k' + '_r.txt.gz')
#             print('Cache deleted: ' + cache_folder + suffix + '_k' + '_r.txt.gz')
#             os.remove(cache_folder + suffix + '_k' + '_i.txt.gz')
#             print('Cache deleted: ' + cache_folder + suffix + '_k' + '_i.txt.gz')
