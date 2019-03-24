"""
This module contains 2 function for splitting a dataset into train/validation
sets for later hyper-parameter selection. The 2 functions are the following:
    1) split_dsets_trainval: Splits the dataset in training/validation sets and
    saves the indexes in selected directory
    2) load_dsets_trainval: Loads the train/validation indexes of the dataset
"""

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import argparse


def split_dsets_trainval(hdf5_file, save_dir):
    """
    This function saves the train and validation indexes of the hdf5_file in
    numpy arrays(binary .npz format).

    Keyword arguments:

    hdf5_file -- hdf5 dataset file(already open)
    save_dir -- directory to save the splits

    Return:
    --
    """
    idx = range(hdf5_file['train']['depth_normalized'].shape[0])
    idx_train, idx_test = train_test_split(idx, random_state=10, test_size=0.5)

    np.savez(save_dir, idx_train, idx_test)


def load_dsets_trainval(train_val_dir):
    """
    This function loads the train/validation indexes of a dataset.

    Keyword arguments:

    train_val_dir -- directory of saved train/validation indexes(.npz format)

    Return:

    idx_train -- indexes of training set
    idx_val -- indexes of validation set
    """
    npzfile = np.load(train_val_dir)
    idx_train = npzfile['arr_0']
    idx_val = npzfile['arr_1']

    return idx_train, idx_val


# If this file is called directly from command line the following lines will
# save NYU and ICVL datasets' train/validation indexes

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='''Provides a split of dataset's indices into training/validation,
        to be used by the batch generator''')
    parser.add_argument('dataset_dir', help='Dataset\'s (in HDF5 format) directory')

    args = parser.parse_args()

    dataset_hdf5 = h5py.File(args.dataset_dir, 'r')
    split_dsets_trainval(dataset_hdf5, './train_test_splits/nyu_split_fuse.npz')
    dataset_hdf5.close()
    """
    icvl_hdf5 = h5py.File('/project/kakadiaris/biometrics/'
                          'shared_datasets/hands_hdf5/ICVL.hdf5', 'r')
    split_dsets_trainval(icvl_hdf5, './train_test_splits/icvl_split.npz')
    icvl_hdf5.close()

    msra_hdf5 = h5py.File('/project/kakadiaris/biometrics/'
                          'shared_datasets/hands_hdf5/MSRA.hdf5', 'r')
    split_dsets_trainval(msra_hdf5, './train_test_splits/msra_split.npz')
    msra_hdf5.close()
    """
