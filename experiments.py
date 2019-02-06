import os
import subprocess
import argparse


def create_jobs(input_channels, net_type, fusion_level,
                fusion_type, predef_hp, validate, save_model,
                save_loss, early_stopping, shuffle, dataset, p, weights_dir):
    jobs_dir =\
            '/home/mvrigkas/hand_pose_estimation/jobs'
    if net_type == 'simple':
        if input_channels == 1:
            input_type = 'depth'
        elif input_channels == 4:
            input_type = 'rgb'
        save_dir = '{0:s}/{1:s}/{2:s}/{3:f}'.format(dataset,
                                                    net_type,
                                                    input_type, p)
        job_dir = os.path.join(jobs_dir, save_dir)
    elif net_type == 'fusing':
        save_dir = '{0:s}/{1:s}/{2:s}/{3:d}/{4:f}'.format(dataset,
                                                          net_type,
                                                          fusion_type,
                                                          fusion_level, p)
        job_dir = os.path.join(jobs_dir, save_dir)
    elif net_type == 'dense_fusing':
        save_dir = '{0:s}/{1:s}/{2:s}/{3:d}/{4:f}'.format(dataset,
                                                          net_type,
                                                          fusion_type,
                                                          fusion_level, p)
        job_dir = os.path.join(jobs_dir, save_dir)
    elif net_type == 'score_fusing':
        save_dir = '{0:s}/{1:s}/{2:s}/{3:f}'.format(dataset,
                                                    net_type,
                                                    fusion_type, p)
        job_dir = os.path.join(jobs_dir, save_dir)
    elif net_type == 'input_fusing':
        save_dir = '{0:s}/{1:s}/{2:s}/{3:f}'.format(dataset,
                                                    net_type,
                                                    fusion_type, p)
        job_dir = os.path.join(jobs_dir, save_dir)

    if not os.path.exists(job_dir):
        os.makedirs(job_dir)
    job = os.path.join(job_dir, 'job')
    splitted = save_dir.split('/')
    if len(splitted) == 4:
        jobname = splitted[0] + splitted[1] + splitted[2] + splitted[3]
    elif len(splitted) == 5:
        jobname = splitted[0] + splitted[1] + splitted[2] + splitted[3]\
                + splitted[4]
    else:
        jobname = splitted[0] + splitted[1] + splitted[2]
    with open(job, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#SBATCH -t 24:00:00\n')
        f.write('#SBATCH -N 1 -n 20\n')
        f.write('#SBATCH -J %s' % jobname+'\n')
        f.write('#SBATCH -o %s' % jobname+'.o%j\n')
        f.write('#SBATCH -e %s' % jobname+'.err%j\n')
        f.write('#SBATCH --mail-user=kazakosv90@gmail.com\n')
        f.write('#SBATCH --mail-type=all\n')
        f.write('#SBATCH --mem=64380\n')
        f.write('#SBATCH -p gpu\n')
        f.write('\n')
        f.write('module add python/2.7-gcc\n')
        f.write('source ~/env/bin/activate\n')
        f.write('module add cuda-toolkit/7.5-cudnn-5.1\n')
        f.write('module list\n')
        f.write('\n')
        f.write('hostname\n')
        f.write('mpdboot\n')
        f.write(('python ~/hand_pose_estimation/training_script.py '
                 + '{0:d} {1:s} {2:f} {3:s} {4:s} {5:s} {6:s} {7:s} {8:s} '
                 + '{9:s} {10:s} {11:s}\n').format(
                     input_channels, net_type, p, str(fusion_level),
                     fusion_type, predef_hp, validate, save_model,
                     save_loss, early_stopping, shuffle,
                     weights_dir))
        f.write('mpdallexit\n')
        f.write('deactivate')


def submit_jobs(input_channels, net_type, fusion_level, fusion_type,
                num_of_exper, dataset, p):
    jobs_dir =\
            '/home/mvrigkas/hand_pose_estimation/jobs'
    if net_type == 'simple':
        if input_channels == 1:
            input_type = 'depth'
        elif input_channels == 4:
            input_type = 'rgb'
        save_dir = '{0:s}/{1:s}/{2:s}/{3:f}'.format(dataset,
                                                    net_type,
                                                    input_type, p)
        job_dir = os.path.join(jobs_dir, save_dir)
    elif net_type == 'fusing':
        save_dir = '{0:s}/{1:s}/{2:s}/{3:d}/{4:f}'.format(dataset,
                                                          net_type,
                                                          fusion_type,
                                                          fusion_level, p)
        job_dir = os.path.join(jobs_dir, save_dir)
    elif net_type == 'dense_fusing':
        save_dir = '{0:s}/{1:s}/{2:s}/{3:d}/{4:f}'.format(dataset,
                                                          net_type,
                                                          fusion_type,
                                                          fusion_level, p)
        job_dir = os.path.join(jobs_dir, save_dir)
    elif net_type == 'score_fusing':
        save_dir = '{0:s}/{1:s}/{2:s}/{3:f}'.format(dataset,
                                                    net_type,
                                                    fusion_type, p)
        job_dir = os.path.join(jobs_dir, save_dir)
    elif net_type == 'input_fusing':
        save_dir = '{0:s}/{1:s}/{2:s}/{3:f}'.format(dataset,
                                                    net_type,
                                                    fusion_type, p)
        job_dir = os.path.join(jobs_dir, save_dir)
    for i in range(num_of_exper):
        subprocess.call(['sbatch', job_dir+'/job'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='''Runs training of the ConvNet of your choice. You can
        train a classical ConvNet on depth, RGB or RGB-D data or you
        can train an architecture that fuses ConvNet towers on different
        inputs (RGB and depth).''')
    parser.add_argument('mode', choices=['create', 'submit'], help='specifies'
                        + 'which mode to use, create sbatch files or submit'
                        + 'experiments based on that file')
    parser.add_argument('num_of_exper', type=int, help='number of'
                        + 'experiments. Use it when you are doing experiments'
                        + 'with random hyperparameters for validation.')
    parser.add_argument('input_channels', choices=[1, 4, 5], type=int,
                        help='number of input channels. 1 for depth, 4 for'
                        + 'rgb, 5 for rgbd or fusion')
    parser.add_argument('net_type',
                        choices=['simple', 'fusing', 'dense_fusing',
                                 'score_fusing', 'input_fusing'],
                        help='type of network')
    parser.add_argument('p', type=float, help='dropout probability')
    parser.add_argument('fusion_level', type=int, nargs='?',
                        help='integer that'
                        + 'specifies in which convolutional layer to fuse')
    parser.add_argument('fusion_type', nargs='?',
                        choices=['sum', 'max', 'concat', 'concatconv', 'local'],
                        help='type of fusion')
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
    if args.fusion_level is None:
        fusion_level = ''
    else:
        fusion_level = args.fusion_level
    if args.fusion_type is None:
        fusion_type = ''
    else:
        fusion_type = args.fusion_type
    if args.predef_hp:
        predef_hp = '--predef_hp'
    else:
        predef_hp = ''
    if args.validate:
        validate = '--validate'
    else:
        validate = ''
    if args.save_model:
        save_model = '--save_model'
    else:
        save_model = ''
    if args.save_loss:
        save_loss = '--save_loss'
    else:
        save_loss = ''
    if args.early_stopping:
        early_stopping = '--early_stopping'
    else:
        early_stopping = ''
    if args.shuffle:
        shuffle = '--shuffle'
    else:
        shuffle = ''
    if args.weights_dir is None:
        weights_dir = ''
    else:
        weights_dir = '--weights_dir ' + args.weights_dir

    if args.mode == 'create':
        create_jobs(args.input_channels, args.net_type, fusion_level,
                    fusion_type, predef_hp, validate,
                    save_model, save_loss, early_stopping,
                    shuffle, 'nyu', args.p, weights_dir)
    else:
        submit_jobs(args.input_channels, args.net_type, fusion_level,
                    fusion_type, args.num_of_exper, 'nyu', args.p)

