"""
This module implements the training and testing procedures of the ConvNet. They
are implemented in two different classes: Training and Test class. Each class
implements all the neccessary tools that are needed, such as batch generators,
theano functions compilation etc. A base class, TrainingTesting provides some
general tools that the other two classes are using.
"""
import time
import os
import cPickle as pickle
import h5py
import numpy as np
import theano.tensor as T
import theano
import lasagne
from networks import ConvNet
from saveloadweights import SaveWeights, LoadWeights
from earlystopping import EarlyStopping
from batchgenerators import BatchGenerator
from splitdatasets import load_dsets_trainval
from randomsearch import sample_hyperparams, save_hyperparams


class TrainingTesting(object):
    SIMPLE = 'simple'
    AUTOENCODING = 'autoencoding'
    FUSING = 'fusing'
    DENSE_FUSING = 'dense_fusing'
    SCORE_FUSING = 'score_fusing'
    INPUT_FUSING = 'input_fusing'
    MSRA = 'MSRA'
    NYU = 'NYU'
    ICVL = 'ICVL'

    def __init__(self, dataset_dir, net_specs_dict, model_hp_dict, num_joints, dataset,
                 group, network_type, input_channels=None, fusion_level=None,
                 fusion_type=None):
        self.convnet = ConvNet(net_specs_dict, model_hp_dict, num_joints)
        self._datasets_dir = dataset_dir
        if dataset not in [self.ICVL, self.MSRA, self.NYU]:
            raise ValueError("dataset can take one of the following values:"
                             + " 'MSRA', 'NYU', 'ICVL'")
        self._network_type = network_type
        self._dataset = dataset
        self._group = group
        self._input_channels = input_channels
        self._fusion_level = fusion_level
        self._fusion_type = fusion_type


class Training(TrainingTesting):
    """
    This class implements the training procedure of the convnet
    """

    def __init__(self, dataset_dir, num_joints, dataset, group, network_type, num_epochs,
                 patience, net_specs_dict, model_hp_dict=None,
                 opt_hp_dict=None, validate=True, input_channels=None,
                 fusion_level=None, fusion_type=None, weights_dir=None):
        if model_hp_dict is None and opt_hp_dict is None:
            opt_hp_dict, model_hp_dict = sample_hyperparams([0.001, 0.1], [
             0.5, 1], [
             0.0, 0.1])
            self._save_settings = True
        else:
            self._save_settings = False
        super(Training, self).__init__(
            dataset_dir, net_specs_dict, model_hp_dict, num_joints, dataset, group,
            network_type, input_channels=input_channels,
            fusion_level=fusion_level, fusion_type=fusion_type)
        if network_type not in [self.SIMPLE, self.AUTOENCODING, self.FUSING,
                                self.DENSE_FUSING, self.SCORE_FUSING,
                                self.INPUT_FUSING]:
            raise ValueError("Network types can take one of the following"
                             + " values: 'simple', 'autoencoding', 'fusing',"
                             + " 'dense_fusing', 'score_fusing',"
                             + " input_fusing")
        self._model_hp_dict = model_hp_dict
        self._opt_hp_dict = opt_hp_dict
        self._num_epochs = num_epochs
        self._patience = patience
        if not isinstance(validate, bool):
            raise TypeError('validate should be boolean')
        self._validate = validate
        self._weights_dir = weights_dir
        return

    def _compile_functions(self):
        if self._network_type == self.SIMPLE:
            input_var = T.tensor4('inputs')
        else:
            input_var1 = T.tensor4('inputs_rgb')
            input_var2 = T.tensor4('inputs_depth')
        target_var = T.matrix('targets')
        # bottleneck_W = np.load('nyu_princ_comp_pose.npz')
        # bottleneck_W = bottleneck_W['arr_0']
        lr = theano.shared(np.array(self._opt_hp_dict['lr'],
                                    dtype=theano.config.floatX))
        lr_decay = np.array(0.1, dtype=theano.config.floatX)
        mom = theano.shared(np.array(self._opt_hp_dict['mom'],
                                     dtype=theano.config.floatX))
        print 'Building the ConvNet...\n'
        if self._network_type == self.SIMPLE:
            net = self.convnet.simple_convnet(self._input_channels,
                                              input_var=input_var)
        elif self._network_type == self.FUSING:
            net = self.convnet.fused_convnets(self._fusion_level,
                                              self._fusion_type,
                                              input_var1=input_var1,
                                              input_var2=input_var2,
                                              weights_dir=self._weights_dir)
        elif self._network_type == self.INPUT_FUSING:
            net = self.convnet.input_fused_convnets(self._fusion_type,
                                                    input_var1=input_var1,
                                                    input_var2=input_var2)
        elif self._network_type == self.DENSE_FUSING:
            net = self.convnet.dense_fused_convnets(
                self._fusion_level, self._fusion_type,
                input_var1=input_var1, input_var2=input_var2,
                weights_dir=self._weights_dir)
        elif self._network_type == self.SCORE_FUSING:
            net = self.convnet.score_fused_convnets(
                self._fusion_type, input_var1=input_var1,
                input_var2=input_var2,
                weights_dir=self._weights_dir)
        print 'Compiling theano functions...\n'
        train_pred = lasagne.layers.get_output(net['output'],
                                               deterministic=False)
        val_pred = lasagne.layers.get_output(net['output'], deterministic=True)
        train_loss = lasagne.objectives.squared_error(train_pred, target_var)
        train_loss = 1 / 2.0 * T.mean(T.sum(train_loss, axis=1))
        val_loss = lasagne.objectives.squared_error(val_pred, target_var)
        val_loss = 1 / 2.0 * T.mean(T.sum(val_loss, axis=1))
        params = lasagne.layers.get_all_params(net['output'], trainable=True)
        updates = lasagne.updates.nesterov_momentum(train_loss, params,
                                                    learning_rate=lr,
                                                    momentum=mom)
        if self._network_type == self.SIMPLE:
            fn_train = theano.function([input_var, target_var], [
             train_loss], updates=updates)
            fn_val = theano.function([input_var, target_var], [val_loss])
        else:
            fn_train = theano.function([input_var1, input_var2, target_var], [
             train_loss], updates=updates)
            fn_val = theano.function([input_var1, input_var2, target_var], [
             val_loss])
        return (fn_train, fn_val, net, lr, lr_decay)

    def _training_loop(self, bg_train, bg_val, fn_train, fn_val, lr,
                              lr_decay, sw=None, es=None):
        """
        This function performs the training loop for the case of a simple
        convnet, where the parameters are updated through backprop and the
        training/validation losses are reported.

        Keyword arguments:

        minibatches_train -- batch generator for the training set
        minibatches_val -- batch generator for the validation set
        fn_train -- theano function that perform parameters updated and
                    computes training loss
        fn_val -- theano function that computes validation loss
        lr -- learning rate(theano shared variable)
        lr_decay -- learning rate decay constant(we use constant decay policy)
        sw -- instance of SaveWeights class(default: None)

        """
        training_information = {}
        train_loss_d = []
        val_loss_d = []
        epoch = 0
        if es is not None or sw is not None:
            time_back = 0
        while epoch < self._num_epochs:
            train_loss = 0
            train_batches = 0
            start_time = time.time()
            for batch in bg_train.generate_batches(self._input_channels):
                if self._network_type == self.SIMPLE:
                    X_batch, y_batch = batch
                    loss = fn_train(X_batch, y_batch)
                else:
                    X_batch_rgb, X_batch_depth, y_batch = batch
                    loss = fn_train(X_batch_rgb, X_batch_depth, y_batch)
                train_loss += loss[0]
                train_batches += 1

            train_loss /= train_batches
            train_loss_d.append(train_loss)
            val_loss = 0
            val_batches = 0
            for batch in bg_val.generate_batches(self._input_channels,
                                                 batch_size=1):
                if self._network_type == self.SIMPLE:
                    X_batch, y_batch = batch
                    loss = fn_val(X_batch, y_batch)
                else:
                    X_batch_rgb, X_batch_depth, y_batch = batch
                    loss = fn_val(X_batch_rgb, X_batch_depth, y_batch)
                val_loss += loss[0]
                val_batches += 1

            val_loss /= val_batches
            val_loss_d.append(val_loss)
            print 'Epoch: {0:d}. Completion time:{1:.3f} '.format(
                epoch + 1, time.time() - start_time)
            print 'Train loss: {0:.5f}\t\tValidation loss:{1:.5f}\t\t\
                Ratio(Val/Train): {2:.5f}'.format(train_loss, val_loss,
                                                  val_loss / train_loss)
            print '--------------------------------------------------------'\
                  + '-----------------------------------'
            if sw is not None:
                stop, go_back = sw.early_stopping_with_lr_decay(val_loss,
                                                                epoch, lr,
                                                                time_back)
                if stop and not go_back or epoch == self._num_epochs - 1:
                    sw.save_weights_numpy()
                    break
                if stop and go_back:
                    time_back += 1
                    epoch = sw.best_epoch - 1
            elif es is not None:
                stop, go_back = es.early_stopping_with_lr_decay(val_loss,
                                                                epoch, lr,
                                                                time_back)
                if stop and not go_back or epoch == self._num_epochs - 1:
                    break
                if stop and go_back:
                    time_back += 1
                    epoch = es.best_epoch - 1
            epoch += 1

        training_information['train_loss'] = train_loss_d
        training_information['val_loss'] = val_loss_d
        return training_information

    def train(self, save_model=False, save_loss=False,
                    early_stopping=True, shuffle=False):
        """
        This function performs the training of our ConvNets. It compiles the
        theano functions and performs parameters updates
        (by calling compile_functions), saves several useful
        information during training and stops using early stopping where also
        the model parameters are saved. All the basic components are described
        below as well as their respective modules/functions:
            1) functions compilation: Training.compile_functions(module:
                trainingtesting). Here you can also find optimization details
                such as regularization term in the loss for autoencoder
            2) load/save weights, early stopping: SaveWeights,
            LoadWeights(module: saveloadweights)
            3) networks definitions: module: networks.py. Here you can find
            details related with network design choices as well as
            regularization layers(e.g. dropout) or other techniques such as
            tied weights in the autoencoder.
        """
        dataset = os.path.join(self._datasets_dir, self._dataset)
        dataset += '.hdf5'
        dset = h5py.File(dataset, 'r')

        fn_train, fn_val, net, lr, lr_decay = self._compile_functions()

        if type(save_model) is not bool:
            raise TypeError('save_model should be boolean')
        if save_model:
            models_dir = './models'
            if not os.path.exists(models_dir):
                os.mkdir(models_dir)

            if self._network_type == self.SIMPLE:
                if self._input_channels == 1:
                    input_type = 'depth'
                elif self._input_channels == 4:
                    input_type = 'rgb'
                save_dir = '{0:s}/{1:s}/{2:s}/{3:f}'.format(
                    self._dataset, self._network_type, input_type,
                    self.convnet._model_hp_dict['p'])
                sw = SaveWeights(os.path.join(models_dir, save_dir), net,
                                 self._patience, 'loss')
            elif self._network_type == self.FUSING:
                save_dir = '{0:s}/{1:s}/{2:s}/{3:d}/{4:f}'.format(
                    self._dataset, self._network_type, self._fusion_type,
                    self._fusion_level, self.convnet._model_hp_dict['p'])
                sw = SaveWeights(os.path.join(models_dir, save_dir), net,
                                 self._patience, 'loss')
            elif self._network_type == self.DENSE_FUSING:
                save_dir = '{0:s}/{1:s}/{2:s}/{3:d}/{4:f}'.format(
                    self._dataset, self._network_type, self._fusion_type,
                    self._fusion_level, self.convnet._model_hp_dict['p'])
                sw = SaveWeights(os.path.join(models_dir, save_dir), net,
                                 self._patience, 'loss')
            elif self._network_type == self.SCORE_FUSING:
                save_dir = '{0:s}/{1:s}/{2:s}/{3:f}'.format(
                    self._dataset, self._network_type, self._fusion_type,
                    self.convnet._model_hp_dict['p'])
                sw = SaveWeights(os.path.join(models_dir, save_dir), net,
                                 self._patience, 'loss')
            elif self._network_type == self.INPUT_FUSING:
                save_dir = '{0:s}/{1:s}/{2:s}/{3:f}'.format(
                    self._dataset, self._network_type, self._fusion_type,
                    self.convnet._model_hp_dict['p'])
                sw = SaveWeights(os.path.join(models_dir, save_dir), net,
                                 self._patience, 'loss')
        elif early_stopping:
            es = EarlyStopping(net, self._patience, 'loss')
        if self._validate:
            idx_train, idx_val = load_dsets_trainval(
                './train_test_splits/nyu_split.npz')
            bg_train = BatchGenerator(dset, self._dataset, self._group,
                                      iterable=idx_train, shuffle=shuffle)
            bg_val = BatchGenerator(dset, self._dataset, self._group,
                                    iterable=idx_val, shuffle=shuffle)
        else:
            bg_train = BatchGenerator(dset, self._dataset, self._group,
                                      shuffle=shuffle)
        print 'Training started...\n'
        if save_model:
            training_information = self._training_loop(
                bg_train, bg_val, fn_train, fn_val, lr, lr_decay, sw=sw)
        elif early_stopping:
            training_information = self._training_loop(
                bg_train, bg_val, fn_train, fn_val, lr, lr_decay, es=es)
        else:
            training_information = self._training_loop(
                bg_train, bg_val, fn_train, fn_val, lr, lr_decay)
        if self._save_settings:
            settings_dir = './settings'
            if not os.path.exists(settings_dir):
                os.mkdir(settings_dir)
            val_loss_array = np.array(training_information['val_loss'])
            best_loss = np.amin(val_loss_array)
            if self._network_type == self.SIMPLE:
                if self._input_channels == 1:
                    input_type = 'depth'
                elif self._input_channels == 4:
                    input_type = 'rgb'
                save_dir = '{0:s}/{1:s}/{2:s}/{3:f}'.format(
                    self._dataset, self._network_type, input_type,
                    self.convnet._model_hp_dict['p'])
                save_hyperparams(os.path.join(settings_dir, save_dir),
                                 self._opt_hp_dict, self._model_hp_dict,
                                 best_loss)
            elif self._network_type == self.FUSING:
                save_dir = '{0:s}/{1:s}/{2:s}/{3:d}/{4:f}'.format(
                    self._dataset, self._network_type, self._fusion_type,
                    self._fusion_level, self.convnet._model_hp_dict['p'])
                save_hyperparams(os.path.join(settings_dir, save_dir),
                                 self._opt_hp_dict, self._model_hp_dict,
                                 best_loss)
            elif self._network_type == self.DENSE_FUSING:
                save_dir = '{0:s}/{1:s}/{2:s}/{3:d}/{4:f}'.format(
                    self._dataset, self._network_type, self._fusion_type,
                    self._fusion_level, self.convnet._model_hp_dict['p'])
                save_hyperparams(os.path.join(settings_dir, save_dir),
                                 self._opt_hp_dict, self._model_hp_dict,
                                 best_loss)
            elif self._network_type == self.SCORE_FUSING:
                save_dir = '{0:s}/{1:s}/{2:s}/{3:f}'.format(
                    self._dataset, self._network_type, self._fusion_type,
                    self.convnet._model_hp_dict['p'])
                save_hyperparams(os.path.join(settings_dir, save_dir),
                                 self._opt_hp_dict, self._model_hp_dict,
                                 best_loss)
            elif self._network_type == self.INPUT_FUSING:
                save_dir = '{0:s}/{1:s}/{2:s}/{3:f}'.format(
                    self._dataset, self._network_type, self._fusion_type,
                    self.convnet._model_hp_dict['p'])
                save_hyperparams(os.path.join(settings_dir, save_dir),
                                 self._opt_hp_dict, self._model_hp_dict,
                                 best_loss)
        if save_loss:
            train_val_loss_dir = './train_val_loss'
            if not os.path.exists(train_val_loss_dir):
                os.mkdir(train_val_loss_dir)
            if self._network_type == self.SIMPLE:
                if self._input_channels == 1:
                    input_type = 'depth'
                elif self._input_channels == 4:
                    input_type = 'rgb'
                save_dir = '{0:s}/{1:s}/{2:s}/{3:f}'.format(
                    self._dataset, self._network_type, input_type,
                    self.convnet._model_hp_dict['p'])
                save_dir = os.path.join(train_val_loss_dir, save_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                with open(os.path.join(save_dir, 'train_val_loss.pkl'), 'wb')\
                        as f:
                    pickle.dump(training_information, f,
                                protocol=pickle.HIGHEST_PROTOCOL)
            elif self._network_type == self.FUSING:
                save_dir = '{0:s}/{1:s}/{2:s}/{3:d}/{4:f}'.format(
                    self._dataset, self._network_type, self._fusion_type,
                    self._fusion_level, self.convnet._model_hp_dict['p'])
                save_dir = os.path.join(train_val_loss_dir, save_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                with open(os.path.join(save_dir, 'train_val_loss.pkl'), 'wb')\
                        as f:
                    pickle.dump(training_information, f,
                                protocol=pickle.HIGHEST_PROTOCOL)
            elif self._network_type == self.DENSE_FUSING:
                save_dir = '{0:s}/{1:s}/{2:s}/{3:d}/{4:f}'.format(
                    self._dataset, self._network_type, self._fusion_type,
                    self._fusion_level, self.convnet._model_hp_dict['p'])
                save_dir = os.path.join(train_val_loss_dir, save_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                with open(os.path.join(save_dir, 'train_val_loss.pkl'), 'wb')\
                        as f:
                    pickle.dump(training_information, f,
                                protocol=pickle.HIGHEST_PROTOCOL)
            elif self._network_type == self.SCORE_FUSING:
                save_dir = '{0:s}/{1:s}/{2:s}/{3:f}'.format(
                    self._dataset, self._network_type, self._fusion_type,
                    self.convnet._model_hp_dict['p'])
                save_dir = os.path.join(train_val_loss_dir, save_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                with open(os.path.join(save_dir, 'train_val_loss.pkl'), 'wb')\
                        as f:
                    pickle.dump(training_information, f,
                                protocol=pickle.HIGHEST_PROTOCOL)
            elif self._network_type == self.INPUT_FUSING:
                save_dir = '{0:s}/{1:s}/{2:s}/{3:f}'.format(
                    self._dataset, self._network_type, self._fusion_type,
                    self.convnet._model_hp_dict['p'])
                save_dir = os.path.join(train_val_loss_dir, save_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                with open(os.path.join(save_dir, 'train_val_loss.pkl'), 'wb')\
                        as f:
                    pickle.dump(training_information, f,
                                protocol=pickle.HIGHEST_PROTOCOL)
        return training_information


class Testing(TrainingTesting):

    def __init__(self, net_specs_dict, model_hp_dict, num_joints, dataset,
                 group, network_type, input_channels=None, fusion_level=None,
                 fusion_type=None, score_fusion=None):
        super(Testing, self).__init__(net_specs_dict, model_hp_dict,
                                      num_joints, dataset, group, network_type,
                                      input_channels=input_channels,
                                      fusion_level=fusion_level,
                                      fusion_type=fusion_type)
        self._score_fusion = score_fusion

    def _compile_functions(self, weights_dir):
        if self._network_type == self.SIMPLE:
            input_var = T.tensor4('inputs')
        else:
            input_var1 = T.tensor4('inputs_rgb')
            input_var2 = T.tensor4('inputs_depth')
        # bottleneck_W = np.load('nyu_princ_comp_pose.npz')
        # bottleneck_W = bottleneck_W['arr_0']
        print 'Building the ConvNet...\n'
        if self._network_type == self.SIMPLE:
            net = self.convnet.simple_convnet(self._input_channels,
                                              input_var=input_var)
        elif self._network_type == self.FUSING:
            net = self.convnet.fused_convnets(self._fusion_level,
                                              self._fusion_type,
                                              input_var1=input_var1,
                                              input_var2=input_var2)
        elif self._network_type == self.INPUT_FUSING:
            net = self.convnet.input_fused_convnets(self._fusion_type,
                                                    input_var1=input_var1,
                                                    input_var2=input_var2)
        elif self._network_type == self.DENSE_FUSING:
            net = self.convnet.dense_fused_convnets(
                self._fusion_level, self._fusion_type,
                input_var1=input_var1, input_var2=input_var2)
        elif self._network_type == self.SCORE_FUSING:
            net = self.convnet.score_fused_convnets(
                self._fusion_type, input_var1=input_var1,
                input_var2=input_var2)
        lw = LoadWeights(weights_dir, net)
        lw.load_weights_numpy()
        pred = lasagne.layers.get_output(net['output'], deterministic=True)
        if self._network_type == self.SIMPLE:
            fn_pred = theano.function([input_var], pred)
        else:
            fn_pred = theano.function([input_var1, input_var2], pred)
        return fn_pred

    def predict(self, weights_dir, save_preds=True):
        dataset = os.path.join(self._datasets_dir, self._dataset)
        dataset += '.hdf5'
        dset = h5py.File(dataset, 'r')
        fn_pred = self._compile_functions(weights_dir)
        bg_test = BatchGenerator(dset, self._dataset, self._group)
        predictions = []
        for batch in bg_test.generate_batches(self._input_channels,
                                              batch_size=1):
            if self._network_type == self.SIMPLE:
                X_batch, y_batch = batch
                pred = fn_pred(X_batch)
                pred = np.squeeze(pred)
            else:
                X_batch_rgb, X_batch_depth, y_batch = batch
                pred = fn_pred(X_batch_rgb, X_batch_depth)
                pred = np.squeeze(pred)

            predictions.append(pred)

        predictions = np.array(predictions)
        if save_preds:
            predictions_dir = './predictions'
            if not os.path.exists(predictions_dir):
                os.mkdir(predictions_dir)
            if self._network_type == self.SIMPLE:
                if self._input_channels == 1:
                    input_type = 'depth'
                elif self._input_channels == 4:
                    input_type = 'rgb'
                save_dir = '{0:s}/{1:s}/{2:s}/{3:f}'.format(
                    self._dataset, self._network_type, input_type,
                    self.convnet._model_hp_dict['p'])
                if not os.path.exists(os.path.join(predictions_dir, save_dir)):
                    os.makedirs(os.path.join(predictions_dir, save_dir))
                np.savez(os.path.join(predictions_dir, save_dir,
                                      'predictions.npz'), predictions)
            elif self._network_type == self.FUSING:
                save_dir = '{0:s}/{1:s}/{2:s}/{3:d}/{4:f}'.format(
                    self._dataset, self._network_type, self._fusion_type,
                    self._fusion_level, self.convnet._model_hp_dict['p'])
                if not os.path.exists(os.path.join(predictions_dir, save_dir)):
                    os.makedirs(os.path.join(predictions_dir, save_dir))
                np.savez(os.path.join(predictions_dir, save_dir,
                                      'predictions.npz'), predictions)
            elif self._network_type == self.DENSE_FUSING:
                save_dir = '{0:s}/{1:s}/{2:s}/{3:d}/{4:f}'.format(
                    self._dataset, self._network_type, self._fusion_type,
                    self._fusion_level, self.convnet._model_hp_dict['p'])
                if not os.path.exists(os.path.join(predictions_dir, save_dir)):
                    os.makedirs(os.path.join(predictions_dir, save_dir))
                np.savez(os.path.join(predictions_dir, save_dir,
                                      'predictions.npz'), predictions)
            elif self._network_type == self.SCORE_FUSING:
                save_dir = '{0:s}/{1:s}/{2:s}/{3:f}'.format(
                    self._dataset, self._network_type, self._fusion_type,
                    self.convnet._model_hp_dict['p'])
                if not os.path.exists(os.path.join(predictions_dir, save_dir)):
                    os.makedirs(os.path.join(predictions_dir, save_dir))
                np.savez(os.path.join(predictions_dir, save_dir,
                                      'predictions.npz'), predictions)
            elif self._network_type == self.INPUT_FUSING:
                save_dir = '{0:s}/{1:s}/{2:s}/{3:f}'.format(
                    self._dataset, self._network_type, self._fusion_type,
                    self.convnet._model_hp_dict['p'])
                if not os.path.exists(os.path.join(predictions_dir, save_dir)):
                    os.makedirs(os.path.join(predictions_dir, save_dir))
                np.savez(os.path.join(predictions_dir, save_dir,
                                      'predictions.npz'), predictions)
        else:
            return predictions

    def extract_kernels(self, layer, weights_dir):

        # bottleneck_W = np.load('nyu_princ_comp_pose.npz')
        # bottleneck_W = bottleneck_W['arr_0']
        print 'Building the ConvNet...\n'
        if self._network_type == self.SIMPLE:
            net = self.convnet.simple_convnet(self._input_channels)
        elif self._network_type == self.FUSING:
            net = self.convnet.fused_convnets(self._fusion_level,
                                              self._fusion_type)
        lw = LoadWeights(weights_dir, net)
        lw.load_weights_numpy()
        return net[layer].W
