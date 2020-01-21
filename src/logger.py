import torch
import numpy as np


class Logger(object):

    def __init__(self, **kwargs):
        self.save_folder = kwargs['project_root_dir'] + '/' + kwargs['save_folder']
        self.early_stopping = EarlyStopping(patience=kwargs['patience'], save_folder=self.save_folder)

    def push_early_stopping(self, val_loss, model):
        self.early_stopping(val_loss, model)

    def print_info(self, epoch, train_loss, train_acc, val_loss, val_acc):
        print("\rEpoch: {}, Train Loss: {}, Train Acc: {}, Val Loss: {}, Val Acc: {}".format(epoch + 1,
                                                                                             round(float(train_loss), ndigits=3),
                                                                                             round(float(train_acc), ndigits=3),
                                                                                             round(float(val_loss), ndigits=3),
                                                                                             round(float(val_acc), ndigits=3)))


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, save_folder='./'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_folder = save_folder

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_folder + '/checkpoint.pt')
        self.val_loss_min = val_loss
