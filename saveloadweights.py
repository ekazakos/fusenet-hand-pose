import numpy as np
from lasagne.layers import set_all_param_values
import os
import cPickle as pickle
from earlystopping import EarlyStopping


class SaveWeights(EarlyStopping):

    def __init__(self, weights_dir, net, patience, loss_or_acc, times=5):
        super(SaveWeights, self).__init__(net, patience,
                                          loss_or_acc, times)
        self.weights_dir = weights_dir

    def save_weights_numpy(self):
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
        np.savez(os.path.join(self.weights_dir, 'weights.npz'),
                 *self.best_weights)
        print 'The best accuracy was {} at epoch {}'.format(
            self.best_loss, self.best_epoch)
        print 'Model parameters were saved to '+self.weights_dir

    def save_weights_pickle(self):
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
        with open(os.path.join(self.weights_dir, 'weights.npz'), 'wb') as f:
            pickle.dump(self.best_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
        print 'The best accuracy was {} at epoch {}'.format(
            self.best_loss, self.best_epoch)
        print 'Model parameters were saved to '+self.weights_dir


class LoadWeights(object):

    def __init__(self, weights_dir, net):
        if not (os.path.exists(weights_dir)):
            raise OSError("Directory doesn't exist")
        self.weights_dir = weights_dir
        self.net = net

    def load_weights_numpy(self):
        print 'Loading weights from {0:s}...\n'.format(self.weights_dir)
        with np.load(self.weights_dir) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        print 'Setting the weights to the model...\n'
        set_all_param_values(self.net['output'], param_values, trainable=True)

    def load_weights_pickle(self):

        with open(self.weights_dir, 'rb') as f:
            print 'Loading weights from {0:s}...\n'.format(self.weights_dir)
            param_values = pickle.load(f)
        print 'Setting the weights to the model...\n'
        set_all_param_values(self.net['output'], param_values, trainable=True)
