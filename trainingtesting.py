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

    def compile_functions_simple(self):
        if self._network_type == self.SIMPLE:
            input_var = T.tensor4('inputs')
        else:
            input_var1 = T.tensor4('inputs_rgb')
            input_var2 = T.tensor4('inputs_depth')
        target_var = T.matrix('targets')
        # bottleneck_W = np.load('/project/kakadiaris/biometrics/'
        #                        + 'shared_datasets/hands_hdf5/'
        #                        + 'nyu_princ_comp_pose.npz')
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

    def compile_functions_autoencoding(self, updates_mode):
        input_var = T.tensor4('inputs')
        target_var1 = T.matrix('targets1')
        target_var2 = T.matrix('targets2')
        lr = theano.shared(np.array(self._opt_hp_dict['lr'], dtype=theano.config.floatX))
        lr_decay = np.array(0.1, dtype=theano.config.floatX)
        mom = theano.shared(np.array(self._opt_hp_dict['mom'], dtype=theano.config.floatX))
        print 'Building the ConvNet...\n'
        net = self.convnet.encoding_convnet(input_var)
        print 'Compiling theano functions...\n'
        train_pred_regr, train_pred_rec = lasagne.layers.get_output([
         net['output'], net['decoder']], deterministic=False)
        val_pred_regr, val_pred_rec = lasagne.layers.get_output([
         net['output'], net['decoder']], deterministic=True)
        train_loss_regr = lasagne.objectives.squared_error(train_pred_regr, target_var1)
        train_loss_regr = 1 / 2.0 * T.mean(T.sum(train_loss_regr, axis=1))
        val_loss_regr = lasagne.objectives.squared_error(val_pred_regr, target_var1)
        val_loss_regr = 1 / 2.0 * T.mean(T.sum(val_loss_regr, axis=1))
        output_fc1 = lasagne.layers.get_output(net['fc1'], deterministic=True)
        fn_fc1 = theano.function([input_var], [output_fc1])
        train_loss_rec = lasagne.objectives.squared_error(train_pred_rec, target_var2)
        train_loss_rec = 1 / 2.0 * T.mean(T.sum(train_loss_rec, axis=1))
        val_loss_rec = lasagne.objectives.squared_error(val_pred_rec, target_var2)
        val_loss_rec = 1 / 2.0 * T.mean(T.sum(val_loss_rec, axis=1))
        if updates_mode == 'double':
            params_regr = lasagne.layers.get_all_params(net['output'], trainable=True)
            updates_regr = lasagne.updates.nesterov_momentum(train_loss_regr, params_regr, learning_rate=lr, momentum=mom)
            fn_train_regr = theano.function([input_var, target_var1], [ train_loss_regr], updates=updates_regr)
            fn_val_regr = theano.function([input_var, target_var1], [
             val_loss_regr])
            params_rec = lasagne.layers.get_all_params(net['decoder'], trainable=True)
            updates_rec = lasagne.updates.nesterov_momentum(self._opt_hp_dict['lambda_rec'] * train_loss_rec, params_rec, learning_rate=lr, momentum=mom)
            fn_train_rec = theano.function([input_var, target_var2], [
             train_loss_rec], updates=updates_rec)
            fn_val_rec = theano.function([input_var, target_var2], [
             val_loss_rec])
            return (
             fn_train_regr, fn_val_regr, fn_train_rec, fn_val_rec,
             fn_fc1, net, lr, lr_decay)
        else:
            params_comb = lasagne.layers.get_all_params([net['output'],
             net['decoder']], trainable=True)
            updates_comb = lasagne.updates.nesterov_momentum(train_loss_regr + self._opt_hp_dict['lambda_rec'] * train_loss_rec, params_comb, learning_rate=lr, momentum=mom)
            fn_train_comb = theano.function([input_var, target_var1,
             target_var2], [
             train_loss_regr, train_loss_rec], updates=updates_comb)
            fn_val_comb = theano.function([input_var, target_var1,
             target_var2], [
             val_loss_regr, val_loss_rec])
            return (
             fn_train_comb, fn_val_comb, fn_fc1, net, lr, lr_decay)

    def _training_loop_simple(self, bg_train, bg_val, fn_train, fn_val, lr,
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

    def _training_loop_autoencoding(self, bg_train, bg_val, fn_train_regr, fn_val_regr, fn_fc1, lr, lr_decay, net, fn_train_rec=None, fn_val_rec=None, sw=None, es=None):
        """
        This function performs the training loop for the case of a simple
        convnet, where the parameters are updated through backprop and the
        training/validation losses are reported.
        
        Keyword arguments:
        
        minibatches_train -- batch generator for the training set
        minibatches_val -- batch generator for the validation set
        fn_train_regr -- theano function that perform parameters updates and
                         computes training loss for the regression part of the
                         network
        fn_val_regr -- theano function that computes validation loss for the
                       regression part of the network
        fn_train_rec -- theano function that perform parameters updates and
                         computes training loss for the regression part of the
                         network
        fn_val_rec -- theano function that computes validation loss for the
                       regression part of the network
        lr -- learning rate(theano shared variable)
        lr_decay -- learning rate decay constant(we use constant decay policy)
        save_model -- boolean thatdefines wether to save model parameters
        updates_mode -- specifies how parameters updates are performed. It can
                        take two different values: 'single', 'double'.
                        'single': updates are performed with one
                        forward-backward pass by encapsulating the gradients
                        wrt both losses in one update
                        'double': updates are performed with two
                        forward-backward passes. In the first, the gradients
                        wrt regression loss are backpropagated
                        while in the second one the gradients wrt to
                        autoencoding loss are backpropagated.
        sw -- instance of SaveWeights class(default: None)
        
        """
        training_information = {}
        train_loss_regr_d = []
        val_loss_regr_d = []
        train_loss_rec_d = []
        val_loss_rec_d = []
        for epoch in xrange(self._num_epochs):
            train_loss_regr = 0
            train_loss_rec = 0
            train_batches = 0
            start_time = time.time()
            for batch in bg_train.generate_batches():
                X_batch, y_batch = batch
                fc1_batch = fn_fc1(X_batch)
                if fn_train_rec is None:
                    loss_regr, loss_rec = fn_train_regr(X_batch, y_batch, fc1_batch[0])
                    train_loss_regr += loss_regr
                    train_loss_rec += loss_rec
                else:
                    loss_regr = fn_train_regr(X_batch, y_batch)
                    loss_rec = fn_train_rec(X_batch, fc1_batch[0])
                    train_loss_regr += loss_regr[0]
                    train_loss_rec += loss_rec[0]
                train_batches += 1

            train_loss_regr /= train_batches
            train_loss_rec /= train_batches
            train_loss_regr_d.append(train_loss_regr)
            train_loss_rec_d.append(train_loss_rec)
            val_loss_regr = 0
            val_loss_rec = 0
            val_batches = 0
            for batch in bg_val.generate_batches(batch_size=1):
                X_batch, y_batch = batch
                fc1_batch = fn_fc1(X_batch)
                if fn_val_rec is None:
                    loss_regr, loss_rec = fn_val_regr(X_batch, y_batch, fc1_batch[0])
                    val_loss_regr += loss_regr
                    val_loss_rec += loss_rec
                else:
                    loss_regr = fn_val_regr(X_batch, y_batch)
                    loss_rec = fn_val_rec(X_batch, fc1_batch[0])
                    val_loss_regr += loss_regr[0]
                    val_loss_rec += loss_rec[0]
                val_batches += 1

            val_loss_regr /= val_batches
            val_loss_rec /= val_batches
            val_loss_regr_d.append(val_loss_regr)
            val_loss_rec_d.append(val_loss_rec)
            print 'Epoch: {0:d}. Completion time:{1:.3f} '.format(epoch + 1, time.time() - start_time)
            print 'Train loss regression: {0:.5f}\t\tValidation loss regression :{1:.5f}\t\tRatio(Val/Train): {2:.5f}'.format(train_loss_regr, val_loss_regr, val_loss_regr / train_loss_regr)
            print 'Train loss reconstruction: {0:.5f}\tValidation loss reconstruction: {1:.5f}\t\tRatio(Val/Train): {2:.5f}'.format(train_loss_rec, val_loss_rec, val_loss_rec / train_loss_rec)
            print '-------------------------------------------------------------------------------------------'
            if sw is not None:
                if sw.early_stopping(val_loss_regr, epoch) or epoch == self._num_epochs - 1:
                    sw.save_weights_numpy()
                    break
            elif es is not None:
                if es.early_stopping(val_loss_regr, epoch) or epoch == self._num_epochs - 1:
                    break

        training_information['train_loss_regr'] = train_loss_regr_d
        training_information['val_loss_regr'] = val_loss_regr_d
        training_information['train_loss_rec'] = train_loss_rec_d
        training_information['val_loss_rec'] = val_loss_rec_d
        return training_information

    def train(self, save_model=False, early_stopping=True, updates_mode='single'):
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
        if self._network_type == self.SIMPLE:
            fn_train, fn_val, net, lr, lr_decay = self.compile_functions_simple(self._input_channels)
        elif updates_mode == 'double':
            fn_train_regr, fn_val_regr, fn_train_rec, fn_val_rec, fn_fc1, net, lr, lr_decay = self.compile_functions_autoencoding(updates_mode)
        else:
            fn_train_comb, fn_val_comb, fn_fc1, net, lr, lr_decay = self.compile_functions_autoencoding(updates_mode)
        if type(save_model) is not bool:
            raise TypeError('save_model should be boolean')
        if save_model:
            sw = SaveWeights('./models/{0:s}/{1:s}.pkl'.format(self._dataset, self._network_type), net, self._patience, 'loss')
        elif early_stopping:
            es = EarlyStopping(net, self._patience, 'loss')
        if self._validate:
            if self._dataset == self.NYU:
                idx_train, idx_val = load_dsets_trainval('./train_test_splits/nyu_split.npz')
            elif self._dataset == self.ICVL:
                idx_train, idx_val = load_dsets_trainval('./train_test_splits/icvl_split.npz')
            else:
                idx_train, idx_val = load_dsets_trainval('./train_test_splits/msra_split.npz')
            bg_train = BatchGenerator(dset, self._dataset, self._group, iterable=idx_train)
            bg_val = BatchGenerator(dset, self._dataset, self._group, iterable=idx_val)
        else:
            bg_train = BatchGenerator(dset, self._dataset, self._group)
        print 'Training started...\n'
        if self._network_type is self.SIMPLE:
            if save_model:
                training_information = self._training_loop_simple(bg_train, bg_val, fn_train, fn_val, lr, lr_decay, sw)
            elif early_stopping:
                training_information = self._training_loop_simple(bg_train, bg_val, fn_train, fn_val, lr, lr_decay, es)
            else:
                training_information = self._training_loop_simple(bg_train, bg_val, fn_train, fn_val, lr, lr_decay)
        elif save_model:
            if updates_mode == 'double':
                training_information = self._training_loop_autoencoding(bg_train, bg_val, fn_train_regr, fn_val_regr, fn_fc1, lr, lr_decay, net, fn_train_rec, fn_val_rec, sw)
            else:
                training_information = self._training_loop_autoencoding(bg_train, bg_val, fn_train_comb, fn_val_comb, fn_fc1, lr, lr_decay, net, sw)
        elif early_stopping:
            if updates_mode == 'double':
                training_information = self._training_loop_autoencoding(bg_train, bg_val, fn_train_regr, fn_val_regr, fn_fc1, lr, lr_decay, net, fn_train_rec, fn_val_rec, es)
            else:
                training_information = self._training_loop_autoencoding(bg_train, bg_val, fn_train_comb, fn_val_comb, fn_fc1, lr, lr_decay, net, es)
        elif updates_mode == 'double':
            training_information = self._training_loop_autoencoding(bg_train, bg_val, fn_train_regr, fn_val_regr, fn_fc1, lr, lr_decay, net, fn_train_rec, fn_val_rec)
        else:
            training_information = self._training_loop_autoencoding(bg_train, bg_val, fn_train_comb, fn_val_comb, fn_fc1, lr, lr_decay, net)
        return training_information

    def train_fused(self, save_model=False, save_loss=False,
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

        fn_train, fn_val, net, lr, lr_decay = self.compile_functions_simple()

        if type(save_model) is not bool:
            raise TypeError('save_model should be boolean')
        if save_model:
            models_dir = '/home/mvrigkas/hand_pose_estimation/models'
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
            training_information = self._training_loop_simple(
                bg_train, bg_val, fn_train, fn_val, lr, lr_decay, sw=sw)
        elif early_stopping:
            training_information = self._training_loop_simple(
                bg_train, bg_val, fn_train, fn_val, lr, lr_decay, es=es)
        else:
            training_information = self._training_loop_simple(
                bg_train, bg_val, fn_train, fn_val, lr, lr_decay)
        if self._save_settings:
            settings_dir = '/home/mvrigkas/hand_pose_estimation/'\
                    + 'settings'
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
            train_val_loss_dir = '/home/mvrigkas/hand_pose_estimation'\
                '/train_val_loss'
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
        # bottleneck_W = np.load('/project/kakadiaris/biometrics/'
        #                        + 'shared_datasets/hands_hdf5/'
        #                        + 'nyu_princ_comp_pose.npz')
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
            predictions_dir = '/home/mvrigkas/hand_pose_estimation/'\
                    + 'predictions'
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

        # bottleneck_W = np.load('/project/kakadiaris/biometrics/' + 'shared_datasets/hands_hdf5/' + 'nyu_princ_comp_pose.npz')
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



