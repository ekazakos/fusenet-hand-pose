from lasagne.layers import get_all_param_values, set_all_param_values


class EarlyStopping(object):

    ACCURACY = 'acc'
    LOSS = 'loss'

    def __init__(self, net, patience, loss_or_acc, times=5):
        self.patience = patience
        if loss_or_acc not in [self.ACCURACY, self.LOSS]:
            raise ValueError('loss_or_acc should take one of the following\
                             values: \'loss\', \'acc\'')
        self.loss_or_acc = loss_or_acc
        self.best_acc = 0
        self.best_loss = float('Inf')
        self.best_epoch = 0
        self.best_weights = None
        self.net = net
        self.times = times

    def early_stopping(self, current_val, current_epoch):
        if self.loss_or_acc == self.ACCURACY:
            if current_val > self.best_acc:
                self.best_acc = current_val
                self.best_epoch = current_epoch
                self.best_weights = get_all_param_values(self.net['output'],
                                                         trainable=True)
                return False
            elif self.best_epoch + self.patience < current_epoch:
                print 'Early Stopping...'
                return True
        else:
            if current_val < self.best_loss:
                self.best_loss = current_val
                self.best_epoch = current_epoch
                self.best_weights = get_all_param_values(self.net['output'],
                                                         trainable=True)
                return False
            elif self.best_epoch + self.patience < current_epoch:
                print 'Early Stopping...'
                return True

    def early_stopping_with_lr_decay(self, current_val, current_epoch, lr,
                                     time):
        if self.loss_or_acc == self.ACCURACY:
            if current_val > self.best_acc:
                self.best_acc = current_val
                self.best_epoch = current_epoch
                self.best_weights = get_all_param_values(self.net['output'],
                                                         trainable=True)
                return False, False
            elif self.best_epoch + self.patience < current_epoch:
                if time < self.times:
                    lr.set_value(lr.get_value()*0.5)
                    set_all_param_values(self.net['output'], self.best_weights,
                                         trainable=True)
                    return True, True
                else:
                    print 'Early Stopping...'
                    return True, False
        else:
            if current_val < self.best_loss:
                self.best_loss = current_val
                self.best_epoch = current_epoch
                self.best_weights = get_all_param_values(self.net['output'],
                                                         trainable=True)
                return False, False
            elif self.best_epoch + self.patience < current_epoch:
                if time < self.times:
                    lr.set_value(lr.get_value()*0.5)
                    set_all_param_values(self.net['output'], self.best_weights,
                                         trainable=True)
                    return True, True
                else:
                    print 'Early Stopping...'
                    return True, False
            else:
                return False, True
