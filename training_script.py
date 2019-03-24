import argparse
import trainingtesting


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Runs training of the ConvNet of your choice. You can
        train a classical ConvNet on depth, RGB or RGB-D data or you
        can train an architecture that fuses ConvNet towers on different
        inputs (RGB and depth).''')
    parser.add_argument('input_channels', choices=[1, 4, 5], type=int,
                        help='number of input'
                        + 'channels. 1 for depth, 4 for rgb, 5 for rgbd or'
                        + 'fusion')
    parser.add_argument('net_type',
                        choices=['simple', 'fusing', 'dense_fusing',
                                 'score_fusing', 'input_fusing'],
                        help='type of network')
    parser.add_argument('p', type=float, help='dropout probability')
    parser.add_argument('fusion_level', type=int, nargs='?',
                        help='integer that specifies in which convolutional'
                        + 'layer to fuse')
    parser.add_argument('fusion_type', nargs='?',
                        choices=['sum', 'max', 'concat', 'concatconv',
                                 'local'],
                        help='Fusion functions. Use \'local\' only with score'
                        + 'fusion.')
    parser.add_argument('--dataset_dir')
    parser.add_argument('--predef_hp', action='store_true', help='boolean that'
                        + 'specifies whether or not to use predifined'
                        + 'hyperparams')
    parser.add_argument('--validate', action='store_true', help='boolean that'
                        + 'specifies validation mode or not')
    parser.add_argument('--save_model', action='store_true',
                        help='boolean that specifies whether to save model'
                        + 'params')
    parser.add_argument('--save_loss', action='store_true', help='boolean that'
                        + 'specifies whether to save loss curves')
    parser.add_argument('--early_stopping', action='store_true',
                        help='boolean that specifies whether to perform early'
                        + 'stopping')
    parser.add_argument('--shuffle', action='store_true',
                        help='boolean that specifies whether to shuffle'
                        + 'training data at each epoch')
    parser.add_argument('--weights_dir', help='Directory of saved weights for'
                        + 'resuming training')
    args = parser.parse_args()
    # Depth-Net
    net_specs_dict = {'num_conv_layers': 9, 'num_conv_filters':
                      (32, 32, 64, 64, 128, 128, 128, 128, 128),
                      'conv_filter_size': (3,)*9,
                      'conv_pad': (1,)*9,
                      'num_fc_units': (4096, 4096)}

    if args.predef_hp:
        opt_hp_dict = {'lr': 0.009, 'mom': 0.98}
        model_hp_dict = {'p': args.p}
    else:
        opt_hp_dict = None
        model_hp_dict = None
    tr = trainingtesting.Training(args.dataset_dir, 14, 'NYU', 'train', args.net_type, 50, 5,
                                  net_specs_dict, model_hp_dict=model_hp_dict,
                                  opt_hp_dict=opt_hp_dict,
                                  validate=args.validate,
                                  input_channels=args.input_channels,
                                  fusion_level=args.fusion_level,
                                  fusion_type=args.fusion_type,
                                  weights_dir=args.weights_dir)
    training_inf = tr.train(save_model=args.save_model,
                            save_loss=args.save_loss,
                            early_stopping=args.early_stopping,
                            shuffle=args.shuffle)
