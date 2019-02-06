from trainingtesting import Testing
# from statsmodels.stats.weightstats import ztest
# import numpy as np

net_specs_dict = {'num_conv_layers': 9, 'num_conv_filters':
                  (32, 32, 64, 64, 128, 128, 128, 128, 128),
                  'conv_filter_size': (3,)*9,
                  'conv_pad': (1,)*9,
                  'num_fc_units': (4096, 4096)}

model_hp_dict = {'p': 0.05}
'''
net_specs_dict = {'num_conv_layers': 3, 'num_conv_filters':
                  (8, 8, 8),
                  'conv_filter_size': (5, 5, 3),
                  'conv_pad': (2, 2, 1),
                  'num_fc_units': (1024, 1024)}
'''
test = Testing(net_specs_dict, model_hp_dict, 14, 'NYU', 'test',
               'score_fusing', input_channels=5, fusion_level=7,
               fusion_type='local')

predictions =\
        test.predict('/home/mvrigkas/hand_pose_estimation/'
                     + 'models/NYU/score_fusing/local/0.050000/weights.npz')
'''
kernels_rgb = test.extract_kernels('conv_rgb3', '/home/ekazakos/lasagne/'
                                   + 'hand_pose_estimation/models/NYU/'
                                   + 'fusing/sum/3/weights.npz')

kernels_depth = test.extract_kernels('conv_depth3', '/home/ekazakos/lasagne/'
                                     + 'hand_pose_estimation/models/NYU/'
                                     + 'fusing/sum/3/weights.npz')
# test.fuse_scores()
kernels_rgb, kernels_depth = kernels_rgb.get_value(), kernels_depth.get_value()
kernels_rgb, kernels_depth = np.reshape(kernels_rgb, (64*32, 9)),\
        np.reshape(kernels_depth, (64*32, 9))
norm_rgb, norm_depth = np.linalg.norm(kernels_rgb, axis=1),\
        np.linalg.norm(kernels_depth, axis=1)
print norm_rgb, norm_depth
print ztest(norm_rgb, x2=norm_depth, alternative='smaller')
'''
