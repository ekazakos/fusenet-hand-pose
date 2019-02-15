"""
Reads datasets and save normalized images and annotations to HDF5
"""

from collections import OrderedDict
import os
import struct
import abc
import numpy as np
from scipy import misc
import imageio
import scipy.io as sio
import h5py
import cv2
from imgnormalization import ImgNormalization


"""
superclass: Dataset
sublasses: one for each dataset

functions::
1. save to hdf5(superclass)
2. read from files and folders and load to numpys (return img)(subclass) 
3. normalize img and joints in both uvd and xyz(subclass) (returns everything that will be saved e.g. depth,com,joints)
4. script tha combines everything and finally saves in hdf5(subclass)
"""
#maybe move here xyz_to_uvd blablabla
#in the subclasses you will put the configurations of the datasets (e.g. number of joints, which joints, focal legths etc)


class Dataset(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, fx, fy, px, py, joints_num, groups_list):

        self.fx = fx
        self.fy = fy
        self.px = px
        self.py = py
        self.joints_num = joints_num
        self.groups_list = groups_list
        self.dataset_size = self._get_dataset_size()
        self._in = ImgNormalization(self.fx, self.fy, self.px, self.py, 250.)

    @abc.abstractmethod
    def _get_dataset_size(self):
        """ 
        Abstract class for computing the dataset size. 
        Different implementation in each dataset 
        """

    def initialize_hdf5(self, f):
        """
        Initializes the dataset structure in HDF5 format

        Keyword arguments:
        f -- HDF5 file(already open)

        Return:
        dset -- object for accessing dataset attributes
        """
        grp = {}
        dset = {}
        for g in self.groups_list:
            grp[g] = f.create_group(g)
            dset[g] = {}
        
        for group in grp.keys():
            dset[group]["depth_normalized"] = grp[group].create_dataset("depth_normalized", (self.dataset_size[group], 1, 128, 128), dtype = np.float32)
            dset[group]["com3D"] = grp[group].create_dataset("com3D", (self.dataset_size[group], 3), dtype = np.float32)
            dset[group]["T"] = grp[group].create_dataset("T", (self.dataset_size[group], 3, 3), dtype = np.float32)
            dset[group]["joints"] = grp[group].create_dataset("joints", (self.dataset_size[group], 3*self.joints_num), dtype = np.float32)
            dset[group]["joints_normalized"] = grp[group].create_dataset("joints_normalized", (self.dataset_size[group], 3*self.joints_num), dtype = np.float32)
            dset[group]["joints3D"] = grp[group].create_dataset("joints3D", (self.dataset_size[group], 3*self.joints_num), dtype = np.float32)
            dset[group]["joints3D_normalized"] = grp[group].create_dataset("joints3D_normalized", (self.dataset_size[group], 3*self.joints_num), dtype = np.float32)
            dset[group]["path"] = grp[group].create_dataset("path", (self.dataset_size[group],), dtype = "S72")

        return dset
    @staticmethod
    def save_hdf5(dset, group, index, depth_normalized, com3D, joints, joints_normalized, joints3D, joints3D_normalized, M, path):

        dset[group]["depth_normalized"][index] = depth_normalized
        dset[group]["com3D"][index] = com3D
        dset[group]["T"][index] = M
        dset[group]["joints"][index] = joints
        dset[group]["joints_normalized"][index] = joints_normalized
        dset[group]["joints3D"][index] = joints3D
        dset[group]["joints3D_normalized"][index] = joints3D_normalized
        dset[group]["path"][index] = path

    def transfrom_joints(self, joints, M):
        joints_normalized = np.zeros(joints.shape)

        for joint in range(joints.shape[1]):
            t = self._in.transform_point_2D( joints[:,joint], M)
            joints_normalized[0,joint] = t[0]
            joints_normalized[1,joint] = t[1]
            joints_normalized[2,joint] = joints[2,joint]

        return joints_normalized

class MSRA_Dataset(Dataset):

    def __init__(self, path, save_dir, group_subjects):

        if not (os.path.exists(path)):
            raise OSError("Directory doesn't exist")
        self.path = path
        self.save_dir = save_dir
        if not isinstance(group_subjects, bool):
            raise TypeError('group_subjects should be a boolean')
        self.group_subjects = group_subjects
        if self.group_subjects:
            groups_list = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
        else:
            groups_list = ['train']
        super(MSRA_Dataset, self).__init__(241.42, 241.42, 160., 120., 21, groups_list)

    def load_image(self, img_dir):
        """
        Loads depth image from binary file. Binary file contains only the bounding box
        of hand as well as its coordinates(img_width, img_height, left, top, right, bottom).

        Keyword arguments:
        img_dir -- the directory of the binary file containing the depth image        

        Return:
        depth -- the depth image
        """
        with open(img_dir,'rb') as f:

            bbox_bin = f.read(24) #read first 6 uint32, unit32 = 4bytes, so 4*6 = 24 bytes 
            bbox = struct.unpack('IIIIII',bbox_bin)
            
            f.seek(24) #move to the position of the 7th digit
            img_bin = f.read((bbox[4] - bbox[2])*(bbox[5] - bbox[3])*4)
            img = struct.unpack('f'*(bbox[4] - bbox[2])*(bbox[5] - bbox[3]), img_bin)
            img = np.array(img)
            img = np.reshape(img, (bbox[5] - bbox[3], bbox[4] - bbox[2]))
        depth = np.zeros((240,320)) # create an image filled with background
        depth[bbox[3]:bbox[5],bbox[2]:bbox[4]] = img
        depth_copy = depth.copy()
        depth[depth==0.] = 2000.
        return depth, depth_copy

    def _get_dataset_size(self):

        if self.group_subjects:
            dataset_size = OrderedDict()
            for g in self.groups_list:
                dataset_size[g] = 0
            for path, dirs, files in os.walk(self.path):
                if (not dirs): 
                    joints_dir = os.path.join(path, 'joint.txt')
                    group = os.path.basename(os.path.dirname(os.path.dirname(joints_dir)))
                    with open(joints_dir, 'r') as f:
                        num = f.readline()
                        num = num.rstrip()
                        num = int(num)
                    dataset_size[group]+=num
        else:
            dataset_size = OrderedDict()
            for g in self.groups_list:
                dataset_size[g] = 0
            for path, dirs, files in os.walk(self.path):
                if (not dirs): 
                    joints_dir = os.path.join(path, 'joint.txt')
                    with open(joints_dir, 'r') as f:
                        num = f.readline()
                        num = num.rstrip()
                        num = int(num)
                    dataset_size['train']+=num
        return dataset_size

    def initialize_hdf5(self, f):
        """
        Initializes the dataset structure in HDF5 format

        Keyword arguments:
        f -- HDF5 file(already open)

        Return:
        dset -- object for accessing dataset attributes
        """
        grp = {}
        dset = {}
        for g in self.groups_list:
            grp[g] = f.create_group(g)
            dset[g] = {}
        
        for group in grp.keys():
            dset[group]["depth_normalized"] = grp[group].create_dataset("depth_normalized", (self.dataset_size[group], 1, 128, 128), dtype = np.float32)
            dset[group]["com3D"] = grp[group].create_dataset("com3D", (self.dataset_size[group], 3), dtype = np.float32)
            dset[group]["T"] = grp[group].create_dataset("T", (self.dataset_size[group], 3, 3), dtype = np.float32)
            dset[group]["joints"] = grp[group].create_dataset("joints", (self.dataset_size[group], 3*self.joints_num), dtype = np.float32)
            dset[group]["joints_normalized"] = grp[group].create_dataset("joints_normalized", (self.dataset_size[group], 3*self.joints_num), dtype = np.float32)
            dset[group]["joints3D"] = grp[group].create_dataset("joints3D", (self.dataset_size[group], 3*self.joints_num), dtype = np.float32)
            dset[group]["joints3D_normalized"] = grp[group].create_dataset("joints3D_normalized", (self.dataset_size[group], 3*self.joints_num), dtype = np.float32)
            dset[group]["path"] = grp[group].create_dataset("path", (self.dataset_size[group],), dtype = "S72")
            dset[group]["subject"] = grp[group].create_dataset("subject", (self.dataset_size[group],), dtype = "S2")
        return dset

    @staticmethod
    def save_hdf5(dset, group, index, depth_normalized, com3D, joints, joints_normalized, joints3D, joints3D_normalized, M, path, subject):

        dset[group]["depth_normalized"][index] = depth_normalized
        dset[group]["com3D"][index] = com3D
        dset[group]["T"][index] = M
        dset[group]["joints"][index] = joints
        dset[group]["joints_normalized"][index] = joints_normalized
        dset[group]["joints3D"][index] = joints3D
        dset[group]["joints3D_normalized"][index] = joints3D_normalized
        dset[group]["path"][index] = path
        dset[group]["subject"][index] = subject

    def load_data(self):
        """
        Walks into directories, reads depth images and joints
        and after centering and normalizing both images and joints, saving them in hdf5.
        It also save in hdf5 other useful information, i.e center of mass of hand in 3D,
        the transformation for joints in UVD. Joints are saved in both UVD and XYZ
        (the initial and the normalized versions).

        Keyword arguments:
        --
        Return:
        --
        """
        if (not os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir)

        f = h5py.File(os.path.join(self.save_dir, self.__class__.__name__.split('_')[0]+'.hdf5'), 'w')
        if self.group_subjects:
            dset = super(MSRA_Dataset, self).initialize_hdf5(f)
        else:
            dset = self.initialize_hdf5(f)
        index = OrderedDict()
        for g in self.groups_list:
            index[g] = 0
        for path, dirs, files in os.walk(self.path):
            if (not dirs): 
                bins = [f for f in files if f.split('.')[1] == 'bin']
                bins = sorted(bins, key = lambda fname: fname.split('_')[0])
                joints_dir = os.path.join(path, 'joint.txt')
                joints_list=[]
                with open(joints_dir,'r') as f:
                    for joints_txt in f:
                        joints = joints_txt.split(' ')
                        joints = [float(j.rstrip()) for j in joints]
                        joints_list.append(joints)
                joints3D_array = np.array(joints_list[1:len(joints_list)])
                for i, bin in enumerate(bins):
                    bin_dir = os.path.join(path, bin)
                    depth, depth_copy = self.load_image(bin_dir)
                    com = self._in.calculate_com(depth_copy)  
                    com3D = self._in.uvd_to_xyz( com )
                    depth_crop_scaled, M = self._in.crop_scale_depth(depth, com)
                    joints3D = np.reshape(joints3D_array[i], (self.joints_num, 3))
                    joints3D = np.swapaxes(joints3D, 0, 1)
                    joints3D[2]*=-1
                    joints3D_normalized, depth_normalized = self._in.joints3D_depth_normalization(joints3D, depth_crop_scaled, com3D)
                    joints = self._in.xyz_to_uvd(joints3D)
                    joints_normalized = self.transfrom_joints(joints, M)
                    
                    # Reshape to 3*joints_num
                    joints_res = np.swapaxes(joints, 0, 1)
                    joints_res = np.reshape(joints_res, (3*self.joints_num,))

                    joints3D_res = np.swapaxes(joints3D, 0, 1)
                    joints3D_res = np.reshape(joints3D_res, (3*self.joints_num,))

                    joints_norm_res = np.swapaxes(joints_normalized, 0, 1)
                    joints_norm_res = np.reshape(joints_norm_res, (3*self.joints_num,))

                    joints3D_norm_res = np.swapaxes(joints3D_normalized, 0, 1)
                    joints3D_norm_res = np.reshape(joints3D_norm_res, (3*self.joints_num,))

                    group = os.path.basename(os.path.dirname(path))
                    dpt = np.reshape(depth_normalized, (1, 128, 128))
                    if self.group_subjects:
                        super(MSRA_Dataset, self).save_hdf5(dset, group, index[group], dpt.astype(np.float32), com3D.astype(np.float32), joints_res.astype(np.float32), joints_norm_res.astype(np.float32), joints3D_res.astype(np.float32), joints3D_norm_res.astype(np.float32), M.astype(np.float32), bin_dir)
                        index[group]+=1
                    else:
                        self.save_hdf5(dset, 'train', index['train'], dpt.astype(np.float32), com3D.astype(np.float32), joints_res.astype(np.float32), joints_norm_res.astype(np.float32), joints3D_res.astype(np.float32), joints3D_norm_res.astype(np.float32), M.astype(np.float32), bin_dir, group)
                        index['train']+=1
        f.close()


class ICVL_Dataset(Dataset):
    #TODO: Add function for selecting only the original images
    def __init__(self, path, save_dir):

        if not (os.path.exists(path)):
            raise OSError("Directory doesn't exist")
        self.path = path
        self.save_dir = save_dir
        groups_list = ['train', 'test1', 'test2']
        super(ICVL_Dataset, self).__init__(241.42, 241.42, 160., 120., 16, groups_list)

    def _get_dataset_size(self):

        dataset_size = OrderedDict()
        for g in self.groups_list:
            dataset_size[g] = 0
        subdirs = ['Training/labels.txt', 'Testing/test_seq_1.txt', 'Testing/test_seq_2.txt']
        for grp, dir_ in zip(dataset_size, subdirs):
            labels_dir = os.path.join(self.path, dir_)
            with open(labels_dir, 'r') as f:
                i=0
                for line in f:
                    line_split = line.split(' ', 1)
                    if not os.path.exists(os.path.join(os.path.join(os.path.join(self.path, os.path.dirname(dir_)),'Depth'), line_split[0])):
                        continue
                    i+=1
            dataset_size[grp] = i

        return dataset_size

    def load_data(self):

        if (not os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir)

        f = h5py.File(os.path.join(self.save_dir, self.__class__.__name__.split('_')[0]+'.hdf5'), 'w')
        dset = self.initialize_hdf5(f)
        subdirs = ['Training/labels.txt', 'Testing/test_seq_1.txt', 'Testing/test_seq_2.txt']
        for group, dir_ in zip(self.groups_list, subdirs):
            depth_dir = os.path.join(self.path, os.path.join(os.path.dirname(dir_), 'Depth'))
            labels_dir = os.path.join(self.path, dir_)

            with open(labels_dir, 'r') as f:
                index=0
                for line in f:
                    line_split = line.split(' ', 1)
                    img_dir = os.path.join(depth_dir, line_split[0])
                    if not os.path.exists(img_dir):
                        continue
                    img = misc.imread(img_dir)
                    img = img.astype(np.float32)
                    joints = line_split[1].rstrip()
                    joints = joints.split(' ')
                    joints = np.asarray(joints, dtype = np.float32)
                    joints = np.reshape(joints, (self.joints_num, 3))
                    joints = np.swapaxes(joints,0,1)
                    joints3D = self._in.uvd_to_xyz(joints)
                    depth, M = self._in.crop_scale_depth(img, joints[:,0])
                    joints3D_normalized, depth_normalized = self._in.joints3D_depth_normalization(joints3D, depth, joints3D[:,0])
                    joints_normalized = self.transfrom_joints(joints, M)
                    dpt = np.reshape(depth_normalized, (1, 128, 128))
                    # Reshape to 3*joints_num
                    joints_res = np.swapaxes(joints, 0, 1)
                    joints_res = np.reshape(joints_res, (3*self.joints_num,))

                    joints3D_res = np.swapaxes(joints3D, 0, 1)
                    joints3D_res = np.reshape(joints3D_res, (3*self.joints_num,))

                    joints_norm_res = np.swapaxes(joints_normalized, 0, 1)
                    joints_norm_res = np.reshape(joints_norm_res, (3*self.joints_num,))

                    joints3D_norm_res = np.swapaxes(joints3D_normalized, 0, 1)
                    joints3D_norm_res = np.reshape(joints3D_norm_res, (3*self.joints_num,))

                    self.save_hdf5(dset, group, index, dpt.astype(np.float32), joints3D[:,0].astype(np.float32), joints_res.astype(np.float32), joints_norm_res.astype(np.float32), joints3D_res.astype(np.float32), joints3D_norm_res.astype(np.float32), M.astype(np.float32), img_dir)
                    index+=1
        f.close()

class NYU_Dataset(Dataset):

    def __init__(self, path, save_dir):
        if not (os.path.exists(path)):
            raise OSError("Directory doesn't exist")
        self.path = path
        self.save_dir = save_dir
        groups_list = ['train', 'test']
        super(NYU_Dataset, self).__init__(588.036865, 587.075073, 320., 240., 14, groups_list)
        self.selected_joints = [32, 3, 0, 9, 6, 15, 12, 21, 18, 27, 25, 24, 30, 31]
        self._in = ImgNormalization(self.fx, self.fy, self.px, self.py, 300.)
        self._in1 = ImgNormalization(self.fx, self.fy, self.px, self.py, 300.*0.87)
        self.subject_change = 2440

    def _get_dataset_size(self):

        dataset_size = OrderedDict()
        for g in self.groups_list:
            dataset_size[g] = 0
        for grp in dataset_size:
            dir_ = '{0:s}/{1:s}/{2:s}'.format(self.path, grp, 'joint_data.mat')
            joint_data = sio.loadmat(dir_)
            joints = joint_data['joint_uvd'][0]
            dataset_size[grp] = joints.shape[0]
        return dataset_size

    def initialize_hdf5(self, f):
        """
        Initializes the dataset structure in HDF5 format

        Keyword arguments:
        f -- HDF5 file(already open)

        Return:
        dset -- object for accessing dataset attributes
        """
        grp = {}
        dset = {}
        for g in self.groups_list:
            grp[g] = f.create_group(g)
            dset[g] = {}
        
        for group in grp.keys():
            dset[group]["depth_normalized"] = grp[group].create_dataset("depth_normalized", (self.dataset_size[group], 1, 128, 128), dtype = np.float32)
            dset[group]["rgb_normalized"] = grp[group].create_dataset("rgb_normalized", (self.dataset_size[group], 4, 128, 128), dtype = np.float32)
            dset[group]["com3D"] = grp[group].create_dataset("com3D", (self.dataset_size[group], 3), dtype = np.float32)
            dset[group]["T"] = grp[group].create_dataset("T", (self.dataset_size[group], 3, 3), dtype = np.float32)
            dset[group]["joints"] = grp[group].create_dataset("joints", (self.dataset_size[group], 3*self.joints_num), dtype = np.float32)
            dset[group]["joints_normalized"] = grp[group].create_dataset("joints_normalized", (self.dataset_size[group], 3*self.joints_num), dtype = np.float32)
            dset[group]["joints3D"] = grp[group].create_dataset("joints3D", (self.dataset_size[group], 3*self.joints_num), dtype = np.float32)
            dset[group]["joints3D_normalized"] = grp[group].create_dataset("joints3D_normalized", (self.dataset_size[group], 3*self.joints_num), dtype = np.float32)

        return dset

    @staticmethod
    def save_hdf5(dset, group, index, depth_normalized, rgb_normalized, com3D, joints, joints_normalized, joints3D, joints3D_normalized, M):

        dset[group]["depth_normalized"][index] = depth_normalized
        dset[group]["rgb_normalized"][index] = rgb_normalized
        dset[group]["com3D"][index] = com3D
        dset[group]["T"][index] = M
        dset[group]["joints"][index] = joints
        dset[group]["joints_normalized"][index] = joints_normalized
        dset[group]["joints3D"][index] = joints3D
        dset[group]["joints3D_normalized"][index] = joints3D_normalized

    def load_image(self, img_dir):
        """
        Loads depth image from binary file. Binary file contains only the bounding box
        of hand as well as its coordinates(img_width, img_height, left, top, right, bottom).

        Keyword arguments:
        img_dir -- the directory of the binary file containing the depth image        

        Return:
        depth -- the depth image
        """

        img = imageio.imread(img_dir)
        _, g, b = np.split(img, 3, axis=2)
        g = np.squeeze(g)
        b = np.squeeze(b)
        g = g.astype(np.int32)
        b = b.astype(np.int32)
        depth = np.bitwise_or(np.left_shift(g,8), b)
        depth = depth.astype(np.float32)

        return depth

    def load_data(self):


        if (not os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir)

        f = h5py.File(os.path.join(self.save_dir, self.__class__.__name__.split('_')[0]+'.hdf5'), 'w')
        dset = self.initialize_hdf5(f)

        for group in self.groups_list:

            labels_dir = '{0:s}/{1:s}/joint_data.mat'.format(self.path, group)
            joint_data = sio.loadmat(labels_dir)
            joints = joint_data['joint_uvd'][0]
            joints3D = joint_data['joint_xyz'][0]
            joints = joints[:, self.selected_joints, :]
            joints3D = joints3D[:, self.selected_joints, :]
            joints = np.swapaxes(joints, 1, 2)
            joints3D = np.swapaxes(joints3D, 1, 2)

            for index in range(joints.shape[0]):
                depth_dir = '{0:s}/{1:s}/depth_1_{2:07d}.png'.format(self.path, group, index+1)
                rgb_dir = '{0:s}/{1:s}/rgb_1_{2:07d}.png'.format(self.path, group, index+1)
                depth = self.load_image(depth_dir)
                rgb = cv2.imread(rgb_dir)
                rgb = rgb[:,:,::-1]
                if group == 'test' and index >= self.subject_change:
                    depth, M = self._in1.crop_scale_depth(depth, joints[index, :, 0])
                    joints3D_normalized, depth_normalized = self._in1.joints3D_depth_normalization(joints3D[index, :, :], depth, joints3D[index, :, 0])
                    rgb, msk = self._in1.crop_scale_rgb(rgb, depth_normalized, joints[index, :, 0])
                    rgb = np.rollaxis(rgb, 2)
                    rgb = rgb.astype(np.float32)
                    msk = msk.astype(np.float32)
                    rgb/=255.
                    msk = np.reshape(msk, (1, 128, 128))
                    rgb = np.vstack((rgb, msk))
                else:
                    depth, M = self._in.crop_scale_depth(depth, joints[index, :, 0])                    
                    joints3D_normalized, depth_normalized = self._in.joints3D_depth_normalization(joints3D[index, :, :], depth, joints3D[index, :, 0])
                    rgb, msk = self._in.crop_scale_rgb(rgb, depth_normalized, joints[index, :, 0])
                    rgb = np.rollaxis(rgb, 2)
                    rgb = rgb.astype(np.float32)
                    msk = msk.astype(np.float32)
                    rgb/=255.
                    msk = np.reshape(msk, (1, 128, 128))
                    rgb = np.vstack((rgb, msk))
                joints_normalized = self.transfrom_joints(joints[index, :, :], M)
                dpt = np.reshape(depth_normalized, (1, 128, 128))
                # Reshape to 3*joints_num
                joints_res = np.swapaxes(joints[index, :, :], 0, 1)
                joints_res = np.reshape(joints_res, (3*self.joints_num,))

                joints3D_res = np.swapaxes(joints3D[index, :, :], 0, 1)
                joints3D_res = np.reshape(joints3D_res, (3*self.joints_num,))

                joints_norm_res = np.swapaxes(joints_normalized, 0, 1)
                joints_norm_res = np.reshape(joints_norm_res, (3*self.joints_num,))

                joints3D_norm_res = np.swapaxes(joints3D_normalized, 0, 1)
                joints3D_norm_res = np.reshape(joints3D_norm_res, (3*self.joints_num,))

                self.save_hdf5(dset, group, index, dpt.astype(np.float32), rgb.astype(np.float32), joints3D[index, :, 0].astype(np.float32), joints_res.astype(np.float32), joints_norm_res.astype(np.float32), joints3D_res.astype(np.float32), joints3D_norm_res.astype(np.float32), M.astype(np.float32))      

        f.close()

    @staticmethod
    def compute_mean_dataset(dataset_dir):
        nyu_dir = os.path.join(dataset_dir, 'NYU.hdf5')
        if not os.path.exists(nyu_dir):
            raise IOError('{0:s} could not be found. Please enter a valid hdf5 file for NYU dataset.'.format(nyu_dir))
        with h5py.File(nyu_dir, 'r') as f:
            mean = np.zeros((3,))
            std = np.zeros((3,))
            N = 0
            # Compute dataset mean
            for i in xrange(f["train/rgb_normalized"].shape[0]):
                r, g, b, m = f["train/rgb_normalized"][i]
                m=m.astype(np.int)
                m=np.bitwise_not(m.astype(np.bool))
                mean[0] += np.sum(r[m])
                mean[1] += np.sum(g[m])
                mean[2] += np.sum(b[m])
                N += np.sum(m.astype(np.int))

            mean/=N
            # Compute dataset standard deviation
            for i in xrange(f["train/rgb_normalized"].shape[0]):
                r, g, b, m = f["train/rgb_normalized"][i]
                m=m.astype(np.int)
                m=np.bitwise_not(m.astype(np.bool))
                std[0] += np.sum(np.square(r[m]-mean[0]))
                std[1] += np.sum(np.square(g[m]-mean[1]))
                std[2] += np.sum(np.square(b[m]-mean[2]))

            std/=N-1

        np.savez(os.path.join(dataset_dir, 'mean_std.npz'), mean, std)

    @staticmethod
    def normalize_dataset(dataset_dir):
        mean_std_hand_dir = os.path.join(dataset_dir, 'mean_std_hand.npz')
        mean_std_bg_dir = os.path.join(dataset_dir, 'mean_std_bg.npz')
        if not (os.path.exists(mean_std_hand_dir)):
            raise IOError('{0:s} could not be found. Please enter a valid file with the mean and the standar deviation of the dataset.'.format(mean_std_hand_dir))
        if not (os.path.exists(mean_std_bg_dir)):
            raise IOError('{0:s} could not be found. Please enter a valid file with the mean and the standar deviation of the dataset.'.format(mean_std_bg_dir))
        mean_std_hand = np.load(os.path.join(mean_std_hand_dir))
        mean_std_bg = np.load(os.path.join(mean_std_bg_dir))
        mean_hand = mean_std_hand['arr_0']
        std_hand = mean_std_hand['arr_1']
        mean_bg = mean_std_bg['arr_0']
        std_bg = mean_std_bg['arr_1']
        nyu_dir = os.path.join(dataset_dir, 'NYU.hdf5')
        if not os.path.exists(nyu_dir):
            raise IOError('{0:s} could not be found. Please enter a valid hdf5 file for NYU dataset.'.format(nyu_dir))
        with h5py.File(nyu_dir, 'r+') as f:
            for i in xrange(f["train/rgb_normalized"].shape[0]):
                rgb = np.rollaxis(f["train/rgb_normalized"][i], 0, start=3)
                rgb[rgb[:,:,3].astype(np.int).astype(np.bool), 0:3] = (rgb[rgb[:,:,3].astype(np.int).astype(np.bool), 0:3] - mean_hand.astype(np.float32)) / np.sqrt(std_hand.astype(np.float32))
                rgb[np.bitwise_not(rgb[:,:,3].astype(np.int).astype(np.bool)), 0:3] = (rgb[np.bitwise_not(rgb[:,:,3].astype(np.int).astype(np.bool)), 0:3] - mean_bg.astype(np.float32)) / np.sqrt(std_bg.astype(np.float32))
                f["train/rgb_normalized"][i] = np.rollaxis(rgb, 2)
            for i in xrange(f["test/rgb_normalized"].shape[0]):
                rgb = np.rollaxis(f["test/rgb_normalized"][i], 0, start=3)
                rgb[rgb[:,:,3].astype(np.int).astype(np.bool), 0:3] = (rgb[rgb[:,:,3].astype(np.int).astype(np.bool), 0:3] - mean_hand.astype(np.float32)) / np.sqrt(std_hand.astype(np.float32))
                rgb[np.bitwise_not(rgb[:,:,3].astype(np.int).astype(np.bool)), 0:3] = (rgb[np.bitwise_not(rgb[:,:,3].astype(np.int).astype(np.bool)), 0:3] - mean_bg.astype(np.float32)) / np.sqrt(std_bg.astype(np.float32))
                f["test/rgb_normalized"][i] = np.rollaxis(rgb, 2)