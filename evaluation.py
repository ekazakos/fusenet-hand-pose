"""
This module implements different evaluation metrics of our hand pose estimation
algorithm given the predictions in the test set. Two famous metrics that are
implemented here are:
    1) Accuracy, that is defined by the fraction of test images that the
    maximum joint error is below a threshold.

    2) Mean error joint, i.e. mean error over the whole test sequence computed
    for each joint separately.
"""

import numpy as np


def accuracy(test_predictions, gt, threshold):
    """
    Computes accuracy of test predictions.

    Keyword arguments:

    test_predictions -- numpy array with predictions of joint positions in the
                        test set
    gt -- ground truth joint positions in the test set
    threshold -- threshold of maximum joint error

    Return:

    acc -- accuracy
    """
    max_error = np.asarray([np.amax(np.linalg.norm(
        gt[i]-test_predictions[i], axis=0))for i in range(gt.shape[0])])
    acc = np.sum((max_error > threshold).astype(dtype=np.int))/gt.shape[0]
    return acc


def mean_joint_error(test_predictions, gt):
    """
    Computes mean joint error of test predictions.

    Keyword arguments:

    test_predictions -- numpy array with predictions of joint positions in the
                        test set
    gt -- ground truth joint positions in the test set

    Return:

    mean_error -- mean error per joint (a numpy array with size same as the
    number of joints)
    """
    mean_error = np.mean(np.asarray(
        [np.linalg.norm(gt[i]-test_predictions[i], axis=0)
         for i in range(gt.shape[0])]), axis=0)

    return mean_error
