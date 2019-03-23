"""
This file contains definitions for different network architectures.
"""
from collections import OrderedDict
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, DenseLayer, dropout, ElemwiseMergeLayer, concat, reshape, Conv1DLayer, ElemwiseSumLayer, LocallyConnected2DLayer
import lasagne.nonlinearities
import lasagne
import theano.tensor as T
from saveloadweights import LoadWeights


class ConvNet(object):
    """
    This class contains all the necessary information for creating a
    network(such as number of layers and number of filters per layer), as well
    as functions that define different networks.
    """
    CONCAT = 'concat'
    CONCATCONV = 'concatconv'
    SUM = 'sum'
    MAX = 'max'
    LOCAL = 'local'

    def __init__(self, net_specs_dict, model_hp_dict, num_joints):

        self._net_specs_dict = net_specs_dict
        self._model_hp_dict = model_hp_dict
        self._num_joints = num_joints

    def simple_convnet(self, input_channels, input_var=None,
                       bottleneck_W=None):
        """
        This is a classical convnet. It contains convolution and
        fully-connected(fc) layers.

        Keyword arguments:
        input_var -- theano variable that specifies the type and dimension of
        the input(default None)

        Return:
        net -- dictionary that contains all the network layers
        """
        net = OrderedDict()
        net['input'] = InputLayer((None, input_channels, 128, 128),
                                  input_var=input_var)
        layer = 0
        for i in range(self._net_specs_dict['num_conv_layers']):
            # Add convolution layers
            net['conv{0:d}'.format(i+1)] = Conv2DLayer(
                net.values()[layer],
                num_filters=self._net_specs_dict['num_conv_filters'][i],
                filter_size=(self._net_specs_dict['conv_filter_size'][i],)*2,
                pad='same')
            layer += 1
            if self._net_specs_dict['num_conv_layers'] <= 2:
                # Add pooling layers
                net['pool{0:d}'.format(i+1)] = MaxPool2DLayer(
                    net.values()[layer], pool_size=(3, 3))
                layer += 1
            else:
                # if i < 2:
                if i < 4:
                    if (i+1) % 2 == 0:
                        # Add pooling layers
                        net['pool{0:d}'.format(i+1)] = MaxPool2DLayer(
                            net.values()[layer], pool_size=(3, 3))
                        layer += 1
                else:
                    '''
                    if (i+1) % 2 == 0:
                        # Add pooling layers
                        net['pool{0:d}'.format(i+1)] = MaxPool2DLayer(
                            net.values()[layer], pool_size=(2, 2))
                        layer += 1
                    '''
                    # if (i+1) == 5 or (i+1) == 8:

                    if (i+1) == 7:
                        # Add pooling layers
                        net['pool{0:d}'.format(i+1)] = MaxPool2DLayer(
                            net.values()[layer], pool_size=(3, 3))
                        layer += 1

        # Add fc-layers
        net['fc1'] = DenseLayer(
            net.values()[layer],
            self._net_specs_dict['num_fc_units'][0])
        # Add dropout layer
        net['dropout1'] = dropout(net['fc1'], p=self._model_hp_dict['p'])
        net['fc2'] = DenseLayer(
            net['dropout1'], self._net_specs_dict['num_fc_units'][1])
        # Add dropout layer
        net['dropout2'] = dropout(net['fc2'], p=self._model_hp_dict['p'])
        if bottleneck_W is not None:
            # Add bottleneck layer
            net['bottleneck'] = DenseLayer(net['dropout2'], 30)
            # Add output layer(linear activation because it's regression)
            net['output'] = DenseLayer(
                net['bottleneck'], 3*self._num_joints,
                W=bottleneck_W[0:30],
                nonlinearity=lasagne.nonlinearities.tanh)
        else:
            # Add output layer(linear activation because it's regression)
            net['output'] = DenseLayer(
                net['dropout2'], 3*self._num_joints,
                nonlinearity=lasagne.nonlinearities.tanh)
        return net

    def input_fused_convnets(self, fusion_type, input_var1=None,
                             input_var2=None, bottleneck_W=None):
        net = OrderedDict()
        net['input_rgb'] = InputLayer((None, 4, 128, 128),
                                      input_var=input_var1)
        layer = 0
        net['input_depth'] = InputLayer((None, 1, 128, 128),
                                        input_var=input_var2)
        layer += 1

        if fusion_type == self.CONCAT:
            net['merge'] = concat([net['input_rgb'],
                                   net['input_depth']]
                                  )
            layer += 1
        elif fusion_type == self.CONCATCONV:
            net['concat'] = concat(
                [net['input_rgb'], net['input_depth']])
            layer += 1
            net['merge'] = Conv2DLayer(net['concat'],
                                       num_filters=1,
                                       filter_size=(1, 1), nonlinearity=None)
            layer += 1

        for i in range(self._net_specs_dict['num_conv_layers']):
            # Add convolution layers
            net['conv{0:d}'.format(i+1)] = Conv2DLayer(
                net.values()[layer],
                num_filters=self._net_specs_dict['num_conv_filters'][i],
                filter_size=(self._net_specs_dict['conv_filter_size'][i],)*2,
                pad='same')
            layer += 1
            if self._net_specs_dict['num_conv_layers'] <= 2:
                # Add pooling layers
                net['pool{0:d}'.format(i+1)] = MaxPool2DLayer(
                    net.values()[layer], pool_size=(3, 3))
                layer += 1
            else:
                # if i < 2:
                if i < 4:
                    if (i+1) % 2 == 0:
                        # Add pooling layers
                        net['pool{0:d}'.format(i+1)] = MaxPool2DLayer(
                            net.values()[layer], pool_size=(3, 3))
                        layer += 1
                else:
                    '''
                    if (i+1) % 2 == 0:
                        # Add pooling layers
                        net['pool{0:d}'.format(i+1)] = MaxPool2DLayer(
                            net.values()[layer], pool_size=(2, 2))
                        layer += 1
                    '''
                    # if (i+1) == 5 or (i+1) == 8:

                    if (i+1) == 7:
                        # Add pooling layers
                        net['pool{0:d}'.format(i+1)] = MaxPool2DLayer(
                            net.values()[layer], pool_size=(3, 3))
                        layer += 1

        # Add fc-layers
        net['fc1'] = DenseLayer(
            net.values()[layer],
            self._net_specs_dict['num_fc_units'][0])
        # Add dropout layer
        net['dropout1'] = dropout(net['fc1'], p=self._model_hp_dict['p'])
        net['fc2'] = DenseLayer(
            net['dropout1'], self._net_specs_dict['num_fc_units'][1])
        # Add dropout layer
        net['dropout2'] = dropout(net['fc2'], p=self._model_hp_dict['p'])
        if bottleneck_W is not None:
            # Add bottleneck layer
            net['bottleneck'] = DenseLayer(net['dropout2'], 30)
            # Add output layer(linear activation because it's regression)
            net['output'] = DenseLayer(
                net['bottleneck'], 3*self._num_joints,
                W=bottleneck_W[0:30],
                nonlinearity=lasagne.nonlinearities.tanh)
        else:
            # Add output layer(linear activation because it's regression)
            net['output'] = DenseLayer(
                net['dropout2'], 3*self._num_joints,
                nonlinearity=lasagne.nonlinearities.tanh)
        return net

    def dense_fused_convnets(self, fusion_level, fusion_type, input_var1=None,
                             input_var2=None, bottleneck_W=None,
                             weights_dir=None):

        net = OrderedDict()
        net['input_rgb'] = InputLayer((None, 4, 128, 128),
                                      input_var=input_var1)
        layer = 0
        for i in range(self._net_specs_dict['num_conv_layers']):
            # Add convolution layers
            net['conv_rgb{0:d}'.format(i+1)] = Conv2DLayer(
                net.values()[layer],
                num_filters=self._net_specs_dict['num_conv_filters'][i],
                filter_size=(self._net_specs_dict['conv_filter_size'][i],)*2,
                pad='same')
            layer += 1
            if self._net_specs_dict['num_conv_layers'] <= 2:
                # Add pooling layers
                net['pool_rgb{0:d}'.format(i+1)] = MaxPool2DLayer(
                    net.values()[layer], pool_size=(3, 3))
                layer += 1
            else:
                # if i < 2 and i != fusion_level-1:
                if i < 4:
                    if (i+1) % 2 == 0:
                        # Add pooling layers
                        net['pool_rgb{0:d}'.format(i+1)] = MaxPool2DLayer(
                            net.values()[layer], pool_size=(3, 3))
                        layer += 1
                else:
                    '''
                    if (i+1) % 2 == 0:
                        # Add pooling layers
                        net['pool{0:d}'.format(i+1)] = MaxPool2DLayer(
                            net.values()[layer], pool_size=(2, 2))
                        layer += 1
                    '''
                    # if (i+1) == 5 or (i+1) == 8:

                    if (i+1) == 7:
                        # Add pooling layers
                        net['pool_rgb{0:d}'.format(i+1)] = MaxPool2DLayer(
                            net.values()[layer], pool_size=(3, 3))
                        layer += 1
        # Fc-layers
        net['fc1_rgb'] = DenseLayer(
            net.values()[layer],
            self._net_specs_dict['num_fc_units'][0])
        layer += 1
        if fusion_level == 2:
            # Add dropout layer
            net['dropout1_rgb'] = dropout(net['fc1_rgb'],
                                          p=self._model_hp_dict['p'])
            layer += 1
            net['fc2_rgb'] = DenseLayer(
                net['dropout1_rgb'], self._net_specs_dict['num_fc_units'][1])
            layer += 1

        net['input_depth'] = InputLayer((None, 1, 128, 128),
                                        input_var=input_var2)
        layer += 1
        for i in range(self._net_specs_dict['num_conv_layers']):
            # Add convolution layers
            net['conv_depth{0:d}'.format(i+1)] = Conv2DLayer(
                net.values()[layer],
                num_filters=self._net_specs_dict['num_conv_filters'][i],
                filter_size=(self._net_specs_dict['conv_filter_size'][i],)*2,
                pad='same')
            layer += 1
            if self._net_specs_dict['num_conv_layers'] <= 2:
                # Add pooling layers
                net['pool_depth{0:d}'.format(i+1)] = MaxPool2DLayer(
                    net.values()[layer], pool_size=(3, 3))
                layer += 1
            else:
                # if i < 2 and i != fusion_level-1:
                if i < 4:
                    if (i+1) % 2 == 0:
                        # Add pooling layers
                        net['pool_depth{0:d}'.format(i+1)] =\
                            MaxPool2DLayer(net.values()[layer],
                                           pool_size=(3, 3))
                        layer += 1
                else:
                    '''
                    if (i+1) % 2 == 0:
                        # Add pooling layers
                        net['pool{0:d}'.format(i+1)] = MaxPool2DLayer(
                            net.values()[layer], pool_size=(2, 2))
                        layer += 1
                    '''
                    # if (i+1) == 5 or (i+1) == 8:

                    if (i+1) == 7:
                        # Add pooling layers
                        net['pool_depth{0:d}'.format(i+1)] =\
                            MaxPool2DLayer(net.values()[layer],
                                           pool_size=(3, 3))
                        layer += 1
        # Fc-layers
        net['fc1_depth'] = DenseLayer(
            net.values()[layer],
            self._net_specs_dict['num_fc_units'][0])
        layer += 1
        if fusion_level == 2:
            # Add dropout layer
            net['dropout1_depth'] = dropout(net['fc1_depth'],
                                            p=self._model_hp_dict['p'])
            layer += 1
            net['fc2_depth'] = DenseLayer(
                net['dropout1_depth'], self._net_specs_dict['num_fc_units'][1])
            layer += 1

        # Fuse ConvNets by fusion_level and fusion_type
        if fusion_type == self.MAX:
            net['merge'] =\
                ElemwiseMergeLayer([net['fc%i_rgb' % fusion_level],
                                    net['fc%i_depth' % fusion_level]],
                                   T.maximum)
            layer += 1
        elif fusion_type == self.SUM:
            net['merge'] =\
                ElemwiseMergeLayer([net['fc%i_rgb' % fusion_level],
                                    net['fc%i_depth' % fusion_level]],
                                   T.add)
            layer += 1
        elif fusion_type == self.CONCAT:
            net['merge'] = concat([net['fc%i_rgb' % fusion_level],
                                   net['fc%i_depth' % fusion_level]])
            layer += 1
        elif fusion_type == self.CONCATCONV:
            net['fc%i_rgb_res' % fusion_level] =\
                reshape(net['fc%i_rgb' % fusion_level], ([0], 1, [1]))
            layer += 1
            net['fc%i_depth_res' % fusion_level] =\
                reshape(net['fc%i_depth' % fusion_level], ([0], 1, [1]))
            layer += 1
            net['concat'] = concat([net['fc%i_rgb_res' % fusion_level],
                                    net['fc%i_depth_res' % fusion_level]])
            layer += 1
            net['merge_con'] = Conv1DLayer(net['concat'],
                                           num_filters=1,
                                           filter_size=(1,),
                                           nonlinearity=None)
            layer += 1
            net['merge'] = reshape(net['merge_con'], ([0], [2]))
            layer += 1

        if fusion_level == 1:
            # Add dropout layer
            net['dropout1'] = dropout(net['merge'],
                                      p=self._model_hp_dict['p'])
            layer += 1
            net['fc2'] = DenseLayer(
                net['dropout1'], self._net_specs_dict['num_fc_units'][1])
            layer += 1
            # Add dropout layer
            net['dropout2'] = dropout(net['fc2'], p=self._model_hp_dict['p'])
            layer += 1
        else:
            # Add dropout layer
            net['dropout2'] = dropout(net['merge'], p=self._model_hp_dict['p'])
            layer += 1
        # Add output layer(linear activation because it's regression)
        if bottleneck_W is not None:
            # Add bottleneck layer
            net['bottleneck'] = DenseLayer(net['dropout2'], 30)
            # Add output layer(linear activation because it's regression)
            net['output'] = DenseLayer(
                net['bottleneck'], 3*self._num_joints,
                W=bottleneck_W[0:30],
                nonlinearity=lasagne.nonlinearities.tanh)
        else:
            # Add output layer(linear activation because it's regression)
            net['output'] = DenseLayer(
                net['dropout2'], 3*self._num_joints,
                nonlinearity=lasagne.nonlinearities.tanh)
        if weights_dir is not None:
            lw = LoadWeights(weights_dir, net)
            lw.load_weights_numpy()
        return net

    def fused_convnets(self, fusion_level, fusion_type, input_var1=None,
                       input_var2=None, bottleneck_W=None, weights_dir=None):

        net = OrderedDict()
        net['input_rgb'] = InputLayer((None, 4, 128, 128),
                                      input_var=input_var1)
        layer = 0
        for i in range(fusion_level):
            # Add convolution layers
            net['conv_rgb{0:d}'.format(i+1)] = Conv2DLayer(
                net.values()[layer],
                num_filters=self._net_specs_dict['num_conv_filters'][i],
                filter_size=(self._net_specs_dict['conv_filter_size'][i],)*2,
                pad='same')
            layer += 1
            if self._net_specs_dict['num_conv_layers'] <= 2 and\
                    i != fusion_level - 1:
                # Add pooling layers
                net['pool_rgb{0:d}'.format(i+1)] = MaxPool2DLayer(
                    net.values()[layer], pool_size=(3, 3))
                layer += 1
            else:
                # if i < 2 and i != fusion_level-1:
                if i < 4:
                    if (i+1) % 2 == 0 and i != fusion_level-1:
                        # Add pooling layers
                        net['pool_rgb{0:d}'.format(i+1)] = MaxPool2DLayer(
                            net.values()[layer], pool_size=(3, 3))
                        layer += 1
                else:
                    '''
                    if (i+1) % 2 == 0:
                        # Add pooling layers
                        net['pool{0:d}'.format(i+1)] = MaxPool2DLayer(
                            net.values()[layer], pool_size=(2, 2))
                        layer += 1
                    '''
                    # if (i+1) == 5 or (i+1) == 8:

                    if (i+1) == 7 and i != fusion_level-1:
                        # Add pooling layers
                        net['pool_rgb{0:d}'.format(i+1)] = MaxPool2DLayer(
                            net.values()[layer], pool_size=(3, 3))
                        layer += 1

        net['input_depth'] = InputLayer((None, 1, 128, 128),
                                        input_var=input_var2)
        layer += 1
        for i in range(fusion_level):
            # Add convolution layers
            net['conv_depth{0:d}'.format(i+1)] = Conv2DLayer(
                net.values()[layer],
                num_filters=self._net_specs_dict['num_conv_filters'][i],
                filter_size=(self._net_specs_dict['conv_filter_size'][i],)*2,
                pad='same')
            layer += 1
            if self._net_specs_dict['num_conv_layers'] <= 2 and\
                    i != fusion_level - 1:
                # Add pooling layers
                net['pool_depth{0:d}'.format(i+1)] = MaxPool2DLayer(
                    net.values()[layer], pool_size=(3, 3))
                layer += 1
            else:
                # if i < 2 and i != fusion_level-1:
                if i < 4:
                    if (i+1) % 2 == 0 and i != fusion_level-1:
                        # Add pooling layers
                        net['pool_depth{0:d}'.format(i+1)] =\
                            MaxPool2DLayer(net.values()[layer],
                                           pool_size=(3, 3))
                        layer += 1
                else:
                    '''
                    if (i+1) % 2 == 0:
                        # Add pooling layers
                        net['pool{0:d}'.format(i+1)] = MaxPool2DLayer(
                            net.values()[layer], pool_size=(2, 2))
                        layer += 1
                    '''
                    # if (i+1) == 5 or (i+1) == 8:

                    if (i+1) == 7 and i != fusion_level-1:
                        # Add pooling layers
                        net['pool_depth{0:d}'.format(i+1)] =\
                            MaxPool2DLayer(net.values()[layer],
                                           pool_size=(3, 3))
                        layer += 1
        # Fuse ConvNets by fusion_level and fusion_type
        if fusion_type == self.MAX:
            net['merge'] =\
                ElemwiseMergeLayer([net['conv_rgb{0:d}'.format(fusion_level)],
                                    net['conv_depth{0:d}'.format(fusion_level)]
                                    ], T.maximum)
            layer += 1
        elif fusion_type == self.SUM:
            net['merge'] =\
                ElemwiseMergeLayer([net['conv_rgb{0:d}'.format(fusion_level)],
                                    net['conv_depth{0:d}'.format(fusion_level)]
                                    ], T.add)
            layer += 1
        elif fusion_type == self.CONCAT:
            net['merge'] = concat([net['conv_rgb{0:d}'.format(fusion_level)],
                                   net['conv_depth{0:d}'.format(fusion_level)]]
                                  )
            layer += 1
        elif fusion_type == self.CONCATCONV:
            net['concat'] = concat(
                [net['conv_rgb{0:d}'.format(fusion_level)],
                 net['conv_depth{0:d}'.format(fusion_level)]])
            layer += 1
            net['merge'] = Conv2DLayer(net['concat'],
                                       num_filters=self._net_specs_dict[
                                       'num_conv_filters'][fusion_level-1],
                                       filter_size=(1, 1), nonlinearity=None)
            layer += 1
        # Max-pooling to the merged
        if fusion_level in [2, 4, 7]:
            net['pool_merged'] = MaxPool2DLayer(net['merge'], pool_size=(3, 3))
            layer += 1
        # Continue the rest of the convolutional part of the network,
        # if the fusion took place before the last convolutional layer,
        # else just connect the convolutional part with the fully connected
        # part
        if self._net_specs_dict['num_conv_layers'] > fusion_level:
            for i in range(fusion_level,
                           self._net_specs_dict['num_conv_layers']):
                # Add convolution layers
                net['conv_merged{0:d}'.format(i+1)] = Conv2DLayer(
                    net.values()[layer],
                    num_filters=self._net_specs_dict['num_conv_filters'][i],
                    filter_size=(self._net_specs_dict['conv_filter_size'][i],)
                    * 2, pad='same')
                layer += 1
                if self._net_specs_dict['num_conv_layers'] <= 2:
                    # Add pooling layers
                    net['pool_merged{0:d}'.format(i+1)] = MaxPool2DLayer(
                        net.values()[layer], pool_size=(3, 3))
                    layer += 1
                else:
                    # if i < 2:
                    if i < 4:
                        if (i+1) % 2 == 0:
                            # Add pooling layers
                            net['pool_merged{0:d}'.format(i+1)] =\
                                MaxPool2DLayer(net.values()[layer],
                                               pool_size=(3, 3))
                            layer += 1
                    else:
                        '''
                        if (i+1) % 2 == 0:
                            # Add pooling layers
                            net['pool{0:d}'.format(i+1)] = MaxPool2DLayer(
                                net.values()[layer], pool_size=(2, 2))
                            layer += 1
                        '''
                        # if (i+1) == 5 or (i+1) == 8:

                        if (i+1) == 7:
                            # Add pooling layers
                            net['pool_merged{0:d}'.format(i+1)] =\
                                MaxPool2DLayer(net.values()[layer],
                                               pool_size=(3, 3))
                            layer += 1
        # Fc-layers
        net['fc1'] = DenseLayer(
            net.values()[layer],
            self._net_specs_dict['num_fc_units'][0])
        # Add dropout layer
        net['dropout1'] = dropout(net['fc1'], p=self._model_hp_dict['p'])
        net['fc2'] = DenseLayer(
            net['dropout1'], self._net_specs_dict['num_fc_units'][1])
        # Add dropout layer
        net['dropout2'] = dropout(net['fc2'], p=self._model_hp_dict['p'])
        if bottleneck_W is not None:
            # Add bottleneck layer
            net['bottleneck'] = DenseLayer(net['dropout2'], 30)
            # Add output layer(linear activation because it's regression)
            net['output'] = DenseLayer(
                net['bottleneck'], 3*self._num_joints,
                W=bottleneck_W[0:30],
                nonlinearity=lasagne.nonlinearities.tanh)
        else:
            # Add output layer(linear activation because it's regression)
            net['output'] = DenseLayer(
                net['dropout2'], 3*self._num_joints,
                nonlinearity=lasagne.nonlinearities.tanh)
        if weights_dir is not None:
            lw = LoadWeights(weights_dir, net)
            lw.load_weights_numpy()
        return net

    def score_fused_convnets(self, fusion_type, input_var1=None,
                             input_var2=None, weights_dir_depth=None,
                             weights_dir_rgb=None, bottleneck_W=None,
                             weights_dir=None):

        net = OrderedDict()
        rgb_net = self.simple_convnet(4, input_var=input_var1,
                                      bottleneck_W=bottleneck_W)
        depth_net = self.simple_convnet(1, input_var=input_var2,
                                        bottleneck_W=bottleneck_W)
        if weights_dir_depth is not None and weights_dir_rgb is not None:
            lw_depth = LoadWeights(weights_dir_depth, depth_net)
            lw_depth.load_weights_numpy()
            lw_rgb = LoadWeights(weights_dir_rgb, rgb_net)
            lw_rgb.load_weights_numpy()
        if fusion_type == self.LOCAL:
            net['reshape_depth'] = reshape(depth_net['output'],
                                           ([0], 1, 1, [1]))
            net['reshape_rgb'] = reshape(rgb_net['output'],
                                         ([0], 1, 1, [1]))
            net['concat'] = concat([net['reshape_depth'], net['reshape_rgb']])
            net['lcl'] = LocallyConnected2DLayer(net['concat'], 1, (1, 1),
                                                 untie_biases=True,
                                                 nonlinearity=None)
            net['output'] = reshape(net['lcl'], ([0], [3]))
        elif fusion_type == self.SUM:
            net['output'] = ElemwiseSumLayer([depth_net['output'],
                                               rgb_net['output']], coeffs=0.5)

        if weights_dir is not None:
            lw = LoadWeights(weights_dir, net)
            lw.load_weights_numpy()
        return net

    def encoding_convnet(self, input_var=None):
        """
        This function implements an encoding convnet. It embodies a hybrid
        auto-encoder to encode the 1st fc-layer, i.e. to project non-linearly
        the 1st fc-layer features.

        Keyword arguments:
        input_var -- theano variable that specifies the type and dimension of
        the input(default None)

        Return:
        net -- dictionary that contains all the network layers
        """
        # TODO: Add dropout only in encoding layer
        net = OrderedDict()
        net['input'] = InputLayer((None, 1, 128, 128), input_var=input_var)
        layer = 0
        for i in range(self._net_specs_dict['num_conv_layers']):
            # Add convolution layers
            net['conv{0:d}'.format(i+1)] = Conv2DLayer(
                net.values()[layer],
                self._net_specs_dict['num_conv_filters'][i],
                (self._net_specs_dict['conv_filter_size'][i],)*2,
                pad=self._net_specs_dict['conv_pad'][i])
            layer += 1
            if self._net_specs_dict['num_conv_layers'] <= 3:
                # Add pooling layers
                net['pool{0:d}'.format(i+1)] = MaxPool2DLayer(
                    net.values()[layer], pool_size=(2, 2))
                layer += 1
            else:
                if i < 2:
                    # Add pooling layers
                    net['pool{0:d}'.format(i+1)] = MaxPool2DLayer(
                        net.values()[layer], pool_size=(2, 2))
                    layer += 1
                else:
                    if (i+1) % 2 == 0:
                        # Add pooling layers
                        net['pool{0:d}'.format(i+1)] = MaxPool2DLayer(
                            net.values()[layer], pool_size=(2, 2))
                        layer += 1
        # Add fc-layers
        net['fc1'] = DenseLayer(
            net.values()[layer],
            self._net_specs_dict['num_fc_units'][0],
            nonlinearity=lasagne.nonlinearities.tanh)

        # Dropout layer
        net['dropout'] = dropout(net['fc1'], p=0.2)

        # Encoding layer
        net['encoder'] = DenseLayer(
            net['dropout'], self._net_specs_dict['num_fc_units'][1],
            nonlinearity=lasagne.nonlinearities.tanh)

        # Add output layer(linear activation because it's regression)
        net['output'] = DenseLayer(
            net['encoder'], 3*self._num_joints,
            nonlinearity=lasagne.nonlinearities.tanh)

        # Add decoding layer
        net['decoder'] = DenseLayer(
            net['encoder'], self._net_specs_dict['num_fc_units'][0],
            W=net['encoder'].W.T, nonlinearity=lasagne.nonlinearities.tanh)
        return net
