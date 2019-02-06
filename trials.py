import trainingtesting


import sys

if sys.argv[1] == 'conv':
    net_specs_dict = {'num_conv_layers': 9, 'num_conv_filters':
                      (32, 32, 64, 64, 128, 128, 128, 128, 128),
                      'conv_filter_size': (3,)*9,
                      'conv_pad': (1,)*9,
                      'num_fc_units': (4096, 4096)}
    opt_hp_dict = {'lr': 0.009, 'mom': 0.98}
    model_hp_dict = {'p': 0.03}
    tr = trainingtesting.Training(14, 'NYU', 'train', 'simple', 100, 3,
                                  net_specs_dict, model_hp_dict=model_hp_dict,
                                  opt_hp_dict=opt_hp_dict, input_channels=3)

    training_inf = tr.train_fused(early_stopping=True, shuffle=True)
elif sys.argv[1] == 'rec':
    net_specs_dict = {'num_conv_layers': 3, 'num_conv_filters':
                      (32, 64, 128), 'conv_filter_size': (3, 3, 3),
                      'conv_pad': (1, 1, 1), 'num_fc_units': (1024, 128)}
    hp_specs_dict = {'lr': 0.01, 'mom': 0.9, 'lambda_con': 0.001,
                     'lambda_rec': 0.01}
    tr = trainingtesting.Training(net_specs_dict, hp_specs_dict, 14, 'NYU',
                                  'train', 'autoencoding', 100, 20)

    training_inf = tr.train(early_stopping=False, updates_mode='double')
elif sys.argv[1] == 'fuse':
    net_specs_dict = {'num_conv_layers': 4, 'num_conv_filters':
                      (32, 64, 128, 128),
                      'conv_filter_size': (3,)*4,
                      'conv_pad': (1,)*4,
                      'num_fc_units': (2048, 2048)}
    hp_specs_dict = {'lr': 0.01, 'mom': 0.9}
    tr = trainingtesting.Training(net_specs_dict, hp_specs_dict, 14, 'NYU',
                                  'train', 'fusing', 100, 20, input_channels=4,
                                  fusion_level=4, fusion_type='concatconv')

    training_inf = tr.train_fused(early_stopping=False)
elif sys.argv[1] == 'dense_fuse':
    net_specs_dict = {'num_conv_layers': 4, 'num_conv_filters':
                      (32, 64, 128, 128),
                      'conv_filter_size': (3,)*4,
                      'conv_pad': (1,)*4,
                      'num_fc_units': (4096, 4096)}
    opt_hp_dict = {'lr': 0.01, 'mom': 0.9}
    model_hp_dict = {'p': 0.03}
    tr = trainingtesting.Training(14, 'NYU', 'train', 'dense_fusing', 100, 20,
                                  net_specs_dict, model_hp_dict=model_hp_dict,
                                  opt_hp_dict=opt_hp_dict, input_channels=4,
                                  fusion_type='concat')

    training_inf = tr.train_fused(early_stopping=False)
