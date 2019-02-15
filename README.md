##  Two-stream convolutional networks for fusing RGB and depth images for hand pose estimation using Lasagne

This is an implementation of the paper [On the Fusion of RGB and Depth Information for Hand Pose Estimation](https://ieeexplore.ieee.org/document/8451022). The code is written in python
using the [Lasagne](https://lasagne.readthedocs.io/en/latest/) DL framework.


## Dataset

Download the [NYU dataset](https://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm#download) and unzip it.
The code is designed to process the data in HDF5 format using [h5py](https://www.h5py.org). To convert
the dataset in HDF5 format run the following code in your terminal:

```python
from datasets_preprocessing.datasets import NYU_Dataset
nyu = NYU_Dataset('/path/NYU/dataset', '/path/NYU/hdf5')
nyu.load_data()

```
where */path/NYU/* should be replaced with the location of the unziped file from above. In ```datasets_preprocessing.datasets```, there are also classes for converting to HDF5 the [ICVL]() and [MSRA]() datasets. Only NYU contains RGB-D images, while 
ICVL and MSRA contain only depth images, so experiments have been done only for NYU. Nevertheless, you may
want to train just the depth stream for ICVL and MSRA.

## Publication

Please reference this publication if you find this code useful:

```
@inproceedings{kazakos_fusion_icip2018, 
    author={E. Kazakos and C. Nikou and I. A. Kakadiaris}, 
    booktitle={25th IEEE International Conference on Image Processing (ICIP)}, 
    title={On the Fusion of RGB and Depth Information for Hand Pose Estimation}, 
    year={2018}, 
    pages={868-872}, 
    month={Oct},
}
```

## Citations

* J. Tompson, M. Stein, Y. LeCun, and K. Perlin, “Real- Time Continuous Pose Recovery of Human Hands Using Convolutional Networks,” ACM Transactions on Graphics, vol. 33, pp. 169:1–169:10, 2014.

