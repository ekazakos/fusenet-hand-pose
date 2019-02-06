import numpy as np
"""
This module implements different minibatch generators, depending on the
dataset. It contains a class namely BatchGenerator which contains functions for
generating minibatches for each dataset.
"""


class BatchGenerator(object):
    """
    This class handles the minibatch generators for each dataset. It contains
    the following functions:
        1) __init__: class constructor
        2) generate_batches: batch generator for NYU and ICVL datasets
        3) generate_batches_msra: batch generator for MSRA dataset
        3) minibatches: returns the correct batch generator depending on the
        dataset
    """
    # Dastasets
    MSRA = 'MSRA'
    NYU = 'NYU'
    ICVL = 'ICVL'

    def __init__(self, hdf5_file, dataset, group, iterable=None,
                 shuffle=False):
        """
        Class constructor. It contains the following fields:
            1) _hdf5_file: hdf5 file of the dataset
            2) _dataset: the name of the dataset(available: "MSRA", "ICVL",
            "NYU")
            3) _group: which group of the _hdf5_file while be iterated. For
            ICVL and NYU if group='train' you have also to specify
            _iterable(see below).For MSRA _group defines the subject that will
            be kept as test set.
            4) _dataset_size: the size of the dataset
            5) _iterable: iterable with ids that specify part of the group to
            be iterated(if you splitted training set to train/validation sets
            provide one iterable with the ids of the training data and one with
            ids of validation data. When group='test' leave it None)
        """
        self._hdf5_file = hdf5_file
        if dataset not in [self.MSRA, self.NYU, self.ICVL]:
            raise ValueError('dataset can take on of the following values:\
                             \'MSRA\', \'ICVL\', \'NYU\'')
        self._dataset = dataset
        self._iterable = iterable
        if group not in self._hdf5_file.keys():
            raise ValueError('group should take one of the following values:\
                             {0:s}'.format(self._hdf5_file.keys()))
        self._group = group
        if self._iterable is not None:
            self._dataset_size = self._iterable.shape[0]
        else:
            self._dataset_size = self._hdf5_file[
                self._group]["depth_normalized"].shape[0]
        self._shuffle = shuffle

    def generate_batches(self, input_channels, batch_size=64):
        start_id = 0
        if self._iterable is None:
            indices = range(self._dataset_size)
        if self._shuffle:
            if self._iterable is not None:
                np.random.shuffle(self._iterable)
            else:
                np.random.shuffle(indices)
        while(start_id < self._dataset_size):
            if self._iterable is not None:
                chunk = slice(start_id, start_id+batch_size)
                chunk = self._iterable[chunk].tolist()
                chunk.sort()
            else:
                chunk = slice(start_id, start_id+batch_size)
                chunk = indices[chunk]
            start_id += batch_size
            if input_channels == 1:
                yield self._hdf5_file[self._group]["depth_normalized"][chunk],\
                    self._hdf5_file[self._group]["joints3D_normalized"][chunk]
            elif input_channels == 4:
                yield self._hdf5_file[self._group]["rgb_normalized"][chunk],\
                    self._hdf5_file[self._group]["joints3D_normalized"][chunk]
            elif input_channels == 5:
                yield self._hdf5_file[self._group]["rgb_normalized"][chunk],\
                    self._hdf5_file[self._group]["depth_normalized"][chunk],\
                    self._hdf5_file[self._group]["joints3D_normalized"][chunk]

    def generate_batches_msra_train(self, batch_size=64):

        groups = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7',
                  'P8'].remove(self._group)
        for grp in groups:
            dsize = self._hdf5_file[grp]["depth_normalized"].shape[0]
            start_id = 0
            while(start_id < dsize):
                chunk = range(start_id, start_id+batch_size)
                start_id += batch_size
                yield self._hdf5_file[grp]["depth_normalized"][chunk],\
                    self._hdf5_file[grp]["joints3D_normalized"][chunk]

    def generate_batches_msra_test(self, batch_size=1):
        start_id = 0
        while(start_id < batch_size):
            chunk = range(start_id, start_id+batch_size)
            start_id += batch_size
            yield self._hdf5_file[self._group]["depth_normalized"][chunk],\
                self._hdf5_file[self._group]["joints3D_normalized"][chunk]
