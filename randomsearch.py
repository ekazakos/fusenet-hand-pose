import cPickle as pickle
import numpy as np
import os


def sample_hyperparams(lr_range, mom_range, p_range):
    opt_hp_dict = {}
    model_hp_dict = {}
    opt_hp_dict['lr'] = 10**(np.random.random() *
                             (np.log10(lr_range[1])-np.log10(lr_range[0])) +
                             np.log10(lr_range[0]))
    opt_hp_dict['mom'] = 10**(np.random.random() *
                              (np.log10(mom_range[1])-np.log(mom_range[0])) +
                              np.log10(mom_range[0]))
    model_hp_dict['p'] = np.random.random() *\
        (p_range[1]-p_range[0]) + p_range[0]

    return opt_hp_dict, model_hp_dict


def save_hyperparams(save_dir, opt_hp_dict, model_hp_dict, best_loss):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # num_files = len([f for f in os.listdir(save_dir) if
    #                os.path.isfile(os.path.join(save_dir, f))])
    hyperparams_file = os.path.join(save_dir,
                                    'setting_lr{0:f}_mom{1:f}_p{2:f}.pkl'.
                                    format(opt_hp_dict['lr'],
                                           opt_hp_dict['mom'],
                                           model_hp_dict['p']))
    with open(hyperparams_file, 'wb') as f:
        pickle.dump(opt_hp_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(model_hp_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(best_loss, f, protocol=pickle.HIGHEST_PROTOCOL)


def find_best_hyperparams(hyperparams_dir):

    loss = []
    hyperparams_file = []

    for f in os.listdir(hyperparams_dir):
        if os.path.isfile(os.path.join(hyperparams_dir, f)):
            with open(os.path.join(hyperparams_dir, f), 'rb') as pf:
                opt_hp_dict = pickle.load(pf)
                model_hp_dict = pickle.load(pf)
                best_loss = pickle.load(pf)
            loss.append(best_loss)
            hyperparams_file.append({'file': f, 'opt_hp': opt_hp_dict,
                                     'model_hp': model_hp_dict})
    ind = np.argsort(np.array(loss))
    for i in ind:
        print 'Loss: {0:f}\tLr: {1:f}\tMom: {2:f}\tP: {3:f}'.format(loss[i],
                                                                    hyperparams_file[i]['opt_hp']['lr'],
                                                                    hyperparams_file[i]['opt_hp']['mom'],
                                                                    hyperparams_file[i]['model_hp']['p'])
    '''
    ind = np.argmin(np.array(loss))
    with open(os.path.join(hyperparams_dir, 'best_setting.txt'), 'w') as f:
        f.write('File: {0:s}\n'.format(hyperparams_file[ind]['file']))
        f.write('Loss: {0:f}\n'.format(loss[ind]))
        f.write('Learning rate: {0:f}\tMomentum: {1:f}\tDropout prob:\
               {2:f}\n'.format(hyperparams_file[ind]['opt_hp']['lr'],
                               hyperparams_file[ind]['opt_hp']['mom'],
                               hyperparams_file[ind]['model_hp']['p']))
    '''
