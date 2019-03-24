from trainingtesting import Testing

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

